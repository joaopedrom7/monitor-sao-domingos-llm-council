# Deploy sem expor API keys

## Streamlit Community Cloud

1. Garante que o repositório não inclui `.streamlit/secrets.toml`.
2. Faz commit da app, `requirements.txt`, `packages.txt` e `.streamlit/secrets.toml.example`.
3. No Streamlit Cloud, cria a app com:
   - Main file path: `app.py`
   - O Streamlit Cloud instala automaticamente as dependências de `requirements.txt`.
4. Em `App settings > Secrets`, adiciona:

```toml
IAEDU_CLAUDE_KEY = "sk-usr-..."
IAEDU_GPT_KEY = "sk-usr-..."
```

## Local

Cria `.streamlit/secrets.toml` a partir do exemplo:

```toml
IAEDU_CLAUDE_KEY = "sk-usr-..."
IAEDU_GPT_KEY = "sk-usr-..."
```

Depois corre:

```bash
streamlit run app.py
```

## Importante

As chaves que estavam hardcoded no ficheiro devem ser revogadas e substituídas por chaves novas, porque já ficaram expostas no código.
