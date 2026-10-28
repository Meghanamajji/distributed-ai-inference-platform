from app.routes import health, metrics_route, model_info, predict

routers = [health.router, model_info.router, predict.router, metrics_route.router]
