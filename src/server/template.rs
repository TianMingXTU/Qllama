//! GGUF `tokenizer.chat_template` 的 Jinja 子集渲染器。
//!
//! 仅解析提示词模板所需的变量插值、消息循环、条件分支和 `|trim`，
//! 不执行任意表达式、不加载模板引擎依赖。任何不支持语法均明确报错，
//! 由调用方选择回退 ChatML 或返回 400/500，避免静默降级。

use std::collections::HashMap;

use serde_json::{json, Value};

/// 模板上下文中的一条对话消息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateMessage {
    pub role: String,
    pub content: String,
}

/// 模板渲染上下文。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateContext {
    pub messages: Vec<TemplateMessage>,
    pub bos_token: String,
    pub eos_token: String,
    pub add_generation_prompt: bool,
}

impl Default for TemplateContext {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            bos_token: String::new(),
            eos_token: String::new(),
            add_generation_prompt: true,
        }
    }
}

/// 缺少 GGUF 模板时使用的 ChatML 降级拼法。
///
/// 这是显式契约：调用方必须在响应和 `/v1/capabilities` 标注
/// `template_fallback: "chatml"`，不得把降级结果称作模型原生模板。
pub fn chatml_fallback(context: &TemplateContext) -> String {
    let mut prompt = String::new();
    for message in &context.messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&message.role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }
    if context.add_generation_prompt {
        prompt.push_str("<|im_start|>assistant\n");
    }
    prompt
}

/// 逐字符解析并渲染模板；支持 `{{ ... }}` 与三类 `{% ... %}` 标签。
///
/// 支持变量：`messages`、`message.role` / `message['role']` /
/// `message.content` / `message['content']`、`loop.last`、`bos_token`、
/// `eos_token`、`add_generation_prompt`；过滤器仅支持 `|trim`。
/// 不支持的标签、变量、过滤器、循环/条件嵌套关系均返回 `Err`。
pub fn render_template(template: &str, context: &TemplateContext) -> Result<String, String> {
    let root = Value::Object(serde_json::Map::from_iter([
        ("bos_token".to_string(), json!(context.bos_token)),
        ("eos_token".to_string(), json!(context.eos_token)),
        (
            "add_generation_prompt".to_string(),
            json!(context.add_generation_prompt),
        ),
        (
            "messages".to_string(),
            Value::Array(
                context
                    .messages
                    .iter()
                    .map(|message| json!({"role": message.role, "content": message.content}))
                    .collect(),
            ),
        ),
    ]));
    let nodes = parse(template)?;
    let mut env = HashMap::new();
    env.insert("bos_token".to_string(), root["bos_token"].clone());
    env.insert("eos_token".to_string(), root["eos_token"].clone());
    env.insert(
        "add_generation_prompt".to_string(),
        root["add_generation_prompt"].clone(),
    );
    env.insert("messages".to_string(), root["messages"].clone());
    let mut output = String::new();
    render_nodes(&nodes, &mut env, &mut output)?;
    Ok(output)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Text(String),
    Expr(String),
    For {
        variable: String,
        iterable: String,
        body: Vec<Node>,
    },
    If {
        condition: String,
        then_body: Vec<Node>,
        else_body: Vec<Node>,
    },
}

fn parse(template: &str) -> Result<Vec<Node>, String> {
    let mut nodes = Vec::new();
    let mut text = String::new();
    let mut rest = template;
    loop {
        let Some(open) = rest.find("{%") else {
            append_expressions(rest, &mut nodes, &mut text)?;
            break;
        };
        append_expressions(&rest[..open], &mut nodes, &mut text)?;
        let after_open = &rest[open + 2..];
        let close = after_open
            .find("%}")
            .ok_or_else(|| "unterminated template tag".to_string())?;
        let tag = after_open[..close].trim();
        rest = &after_open[close + 2..];
        if !text.is_empty() {
            nodes.push(Node::Text(std::mem::take(&mut text)));
        }
        if let Some(declaration) = tag.strip_prefix("for ") {
            let (body, remainder) = parse_block(rest, "for")?;
            let (variable, iterable) = declaration
                .trim()
                .split_once(" in ")
                .ok_or_else(|| "for tag must be 'for x in iterable'".to_string())?;
            nodes.push(Node::For {
                variable: variable.trim().to_string(),
                iterable: iterable.trim().to_string(),
                body,
            });
            rest = remainder;
        } else if let Some(condition) = tag.strip_prefix("if ") {
            let (then_body, else_body, remainder) = parse_if(rest)?;
            nodes.push(Node::If {
                condition: condition.trim().to_string(),
                then_body,
                else_body,
            });
            rest = remainder;
        } else {
            return Err(format!("unsupported template tag: {tag}"));
        }
    }
    if !text.is_empty() {
        nodes.push(Node::Text(text));
    }
    Ok(nodes)
}

fn append_expressions(
    source: &str,
    nodes: &mut Vec<Node>,
    text: &mut String,
) -> Result<(), String> {
    let mut rest = source;
    while let Some(open) = rest.find("{{") {
        text.push_str(&rest[..open]);
        let after_open = &rest[open + 2..];
        let close = after_open
            .find("}}")
            .ok_or_else(|| "unterminated template expression".to_string())?;
        if !text.is_empty() {
            nodes.push(Node::Text(std::mem::take(text)));
        }
        nodes.push(Node::Expr(after_open[..close].trim().to_string()));
        rest = &after_open[close + 2..];
    }
    text.push_str(rest);
    Ok(())
}

fn parse_block<'a>(source: &'a str, kind: &str) -> Result<(Vec<Node>, &'a str), String> {
    let end_tag = format!("end{kind}");
    let mut depth = 1usize;
    let mut offset = 0usize;
    while let Some(open) = source[offset..].find("{%") {
        let absolute = offset + open;
        let after_open = &source[absolute + 2..];
        let close = after_open
            .find("%}")
            .ok_or_else(|| "unterminated template tag".to_string())?;
        let tag = after_open[..close].trim();
        if tag.starts_with(&format!("{kind} ")) {
            depth += 1;
        } else if tag == end_tag {
            depth -= 1;
            if depth == 0 {
                let body = parse(&source[..absolute])?;
                return Ok((body, &after_open[close + 2..]));
            }
        }
        offset = absolute + 2 + close + 2;
    }
    Err(format!("missing {end_tag}"))
}

fn parse_if(source: &str) -> Result<(Vec<Node>, Vec<Node>, &str), String> {
    let mut depth = 1usize;
    let mut else_at = None;
    let mut offset = 0usize;
    while let Some(open) = source[offset..].find("{%") {
        let absolute = offset + open;
        let after_open = &source[absolute + 2..];
        let close = after_open
            .find("%}")
            .ok_or_else(|| "unterminated template tag".to_string())?;
        let tag = after_open[..close].trim();
        if tag.starts_with("if ") {
            depth += 1;
        } else if tag == "endif" {
            depth -= 1;
            if depth == 0 {
                let body_source = &source[..absolute];
                let (then_source, else_source) = match else_at {
                    Some(index) => (&body_source[..index], &body_source[index..]),
                    None => (body_source, ""),
                };
                let else_body = if else_source.is_empty() {
                    Vec::new()
                } else {
                    let trimmed = else_source
                        .strip_prefix("{% else %}")
                        .or_else(|| else_source.strip_prefix("{%else%}"))
                        .unwrap_or(else_source);
                    parse(trimmed)?
                };
                return Ok((parse(then_source)?, else_body, &after_open[close + 2..]));
            }
        } else if depth == 1 && tag == "else" {
            else_at = Some(absolute);
        }
        offset = absolute + 2 + close + 2;
    }
    Err("missing endif".to_string())
}

fn render_nodes(
    nodes: &[Node],
    env: &mut HashMap<String, Value>,
    output: &mut String,
) -> Result<(), String> {
    for node in nodes {
        match node {
            Node::Text(text) => output.push_str(text),
            Node::Expr(expression) => output.push_str(&eval_expression(expression, env)?),
            Node::For {
                variable,
                iterable,
                body,
            } => {
                if iterable != "messages" {
                    return Err(format!("unsupported iterable: {iterable}"));
                }
                let values = env
                    .get("messages")
                    .and_then(Value::as_array)
                    .cloned()
                    .ok_or_else(|| "messages must be an array".to_string())?;
                let total = values.len();
                for (index, value) in values.into_iter().enumerate() {
                    env.insert(variable.clone(), value);
                    env.insert("loop.last".to_string(), json!(index + 1 == total));
                    render_nodes(body, env, output)?;
                }
                env.remove(variable);
                env.remove("loop.last");
            }
            Node::If {
                condition,
                then_body,
                else_body,
            } => {
                let branch = if eval_condition(condition, env)? {
                    then_body
                } else {
                    else_body
                };
                render_nodes(branch, env, output)?;
            }
        }
    }
    Ok(())
}

fn eval_condition(condition: &str, env: &HashMap<String, Value>) -> Result<bool, String> {
    let (negated, path) = match condition.trim().strip_prefix("not ") {
        Some(path) => (true, path.trim()),
        None => (false, condition.trim()),
    };
    let value = lookup(path, env).ok_or_else(|| format!("unknown template variable: {path}"))?;
    let truthy = match value {
        Value::Bool(flag) => *flag,
        Value::Null => false,
        Value::String(text) => !text.is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(map) => !map.is_empty(),
        Value::Number(number) => number.as_f64().unwrap_or(0.0) != 0.0,
    };
    Ok(if negated { !truthy } else { truthy })
}

fn eval_expression(expression: &str, env: &HashMap<String, Value>) -> Result<String, String> {
    let (path, trim) = match expression.strip_suffix("|trim") {
        Some(path) => (path.trim(), true),
        None => (expression.trim(), false),
    };
    let value = lookup(path, env)
        .ok_or_else(|| format!("unknown template variable: {path}"))?
        .clone();
    let mut rendered = match &value {
        Value::String(text) => text.clone(),
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => number.to_string(),
        Value::Null => String::new(),
        _ => return Err(format!("template variable is not scalar: {path}")),
    };
    if trim {
        rendered = rendered.trim().to_string();
    }
    Ok(rendered)
}

fn lookup<'a>(path: &str, env: &'a HashMap<String, Value>) -> Option<&'a Value> {
    if let Some(value) = env.get(path) {
        return Some(value);
    }
    let field_start = path
        .find('[')
        .map(|index| index + 1)
        .or_else(|| path.find('.').map(|index| index + 1))?;
    let base = env.get(&path[..field_start - 1])?;
    let raw_field = &path[field_start..];
    let field = raw_field
        .strip_suffix(']')
        .unwrap_or(raw_field)
        .trim_matches('\'')
        .trim_matches('"');
    base.get(field)
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::{chatml_fallback, render_template, TemplateContext, TemplateMessage};

    fn context() -> TemplateContext {
        TemplateContext {
            messages: vec![
                TemplateMessage {
                    role: "system".into(),
                    content: "You are Qllama.".into(),
                },
                TemplateMessage {
                    role: "user".into(),
                    content: "hello".into(),
                },
            ],
            bos_token: "<s>".into(),
            eos_token: "</s>".into(),
            add_generation_prompt: true,
        }
    }

    #[test]
    fn renders_chatml_template_exactly() {
        let template = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        assert_eq!(
            render_template(template, &context()).unwrap(),
            chatml_fallback(&context())
        );
    }

    #[test]
    fn renders_llama3_and_qwen_tokens() {
        let template = "{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n{{ message['content'] }}<|eot_id|>{% if not loop.last %}\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n{% endif %}";
        let actual = render_template(template, &context()).unwrap();
        assert!(actual.starts_with("<s><|start_header_id|>system"));
        assert!(actual.ends_with("<|start_header_id|>assistant<|end_header_id|>\n"));
        assert!(actual.contains("hello<|eot_id|>"));
        let qwen = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content |trim}}<|im_end|>{% if loop.last %}\n{% endif %}{% endfor %}<|im_start|>assistant\n";
        assert_eq!(render_template(qwen, &context()).unwrap(), "<|im_start|>system\nYou are Qllama.<|im_end|><|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn rejects_unsupported_template_syntax() {
        let error = render_template("{{ unknown }}", &context()).unwrap_err();
        assert!(error.contains("unknown template variable"));
        assert!(render_template("{% set x = 1 %}", &context()).is_err());
        assert!(render_template("{{ bos_token |upper }}", &context()).is_err());
        assert!(render_template("{% for message in other %}{% endfor %}", &context()).is_err());
    }
}
