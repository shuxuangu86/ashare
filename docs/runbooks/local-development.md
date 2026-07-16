# 本地开发运行手册

## 1. Windows / WSL2 准备

在 Docker Desktop 中启用 WSL2 后端和当前 Linux 发行版集成。在 WSL2 终端确认：

```bash
docker version
docker compose version
uv --version
python --version
```

Python 应为 3.11.x。项目的 `requires-python` 暂时兼容 3.12 以便 CI 交叉验证，但本地
标准运行时固定为 3.11。

## 2. 初始化

```bash
cp .env.example .env
```

修改 `.env` 的 `POSTGRES_PASSWORD`。不要把数据源 Token 或券商凭据写入 YAML。

```bash
uv sync --group dev
uv run aquant config-check \
  --config config/base.yaml \
  --config config/data.yaml \
  --config config/backtest.yaml \
  --config config/risk.yaml \
  --config config/brokers/paper.yaml
docker compose config --quiet
docker compose up -d --build
```

## 3. 健康检查

```bash
docker compose ps
docker compose exec postgres pg_isready -U aquant -d aquant
docker compose exec redis redis-cli ping
curl --fail http://127.0.0.1:4200/api/health
curl --fail http://127.0.0.1:5000/health
```

预期所有容器为 `healthy`，Redis 返回 `PONG`，两个 HTTP 检查返回成功。

## 4. 质量门槛

```bash
make quality
```

提交前必须同时通过 Ruff、格式检查、mypy strict、pytest 和 85% 覆盖率门槛。

## 5. 安全停止

```bash
docker compose down
```

此命令保留数据库和制品卷。不要使用 `down -v`，除非已经备份且明确要删除本地状态。

## 6. 常见故障

- 端口占用：在 `.env` 调整对应 `*_HOST_PORT`，不要把服务绑定到公网地址。
- PostgreSQL 初始化脚本未执行：脚本只在新数据卷首次启动时执行；不要随意删除已有卷。
- 环境覆盖未生效：嵌套键使用双下划线，例如 `AQUANT_DATABASE__PORT=55432`。
- 实盘配置被拒绝：这是安全行为；Day 16 以后按审批流程开放，不要修改校验器绕过。
