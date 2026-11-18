# ai-gateway-mcp

Servidor MCP (FastMCP) do OpenPixel. Expõe o `backend-core` como tools para agentes.

Tools: `generate_pixel_art`, `generate_8_directions`, `animate_asset`,
`generate_complete_map` (retorna assets + `manual_markdown`), `export_engine_bundle`.

```
pip install fastmcp
python ai-gateway-mcp/server.py      # transporte stdio
```

O service (`openpixel_core.service`) é importado de forma lazy dentro de cada tool.
