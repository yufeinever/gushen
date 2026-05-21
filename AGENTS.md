# AGENTS.md

## Project Instructions

- 国内数据源默认直连，不主动走代理。包括但不限于 AKShare 调用的东方财富、新浪、巨潮资讯、交易所等国内站点；只有确认直连不通且目标确实需要代理时，才临时设置代理。
- 访问 GitHub、OpenAI、海外文档或其他外网资源时，如果网络不通，可以使用本地代理，例如 `http://127.0.0.1:10808`。
- Windows PowerShell 通过 SSH 给远端传脚本时，中文/emoji 字符串可能被转码破坏；筛选或写入中文必须用 Unicode escape、远端原始值、节点序号或 server/port，不能直接在本地命令里写中文条件。
- 远端更新前后都要查目标行数。
- 项目改动后要提交。
