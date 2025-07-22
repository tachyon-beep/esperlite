# Esper API Documentation

## Overview

The Esper platform provides REST APIs for all major services. Each service runs independently and communicates via well-defined interfaces.

## Service Endpoints

### Urza (Library Service)
- **Base URL**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`

#### Key Endpoints
- `GET /health` - Health check
- `GET /kernels` - List available kernels
- `GET /kernels/{kernel_id}` - Get kernel metadata
- `POST /kernels` - Register new kernel
- `GET /blueprints` - List blueprints
- `POST /blueprints` - Create blueprint

### Tamiyo (Strategic Controller)
- **Base URL**: `http://localhost:8001`
- **API Docs**: `http://localhost:8001/docs`

#### Key Endpoints
- `GET /health` - Health check
- `POST /analyze` - Analyze model state
- `GET /decisions` - Get decision history
- `POST /decisions/evaluate` - Evaluate potential adaptation
- `GET /metrics` - Performance metrics

### Tolaria (Training Orchestrator)
- **Base URL**: `http://localhost:8080`
- **API Docs**: `http://localhost:8080/docs`

#### Key Endpoints
- `GET /health` - Health check
- `POST /training/start` - Start training session
- `GET /training/status` - Get training status
- `POST /training/pause` - Pause training
- `POST /training/resume` - Resume training
- `GET /metrics` - Training metrics

## Authentication

All API endpoints require authentication via API key:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/kernels
```

## Common Response Formats

### Success Response
```json
{
  "status": "success",
  "data": { ... },
  "timestamp": "2024-07-22T10:30:00Z"
}
```

### Error Response
```json
{
  "status": "error",
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Description of the error",
    "details": { ... }
  },
  "timestamp": "2024-07-22T10:30:00Z"
}
```

## Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

## Rate Limiting

- Default: 100 requests per minute per API key
- Burst: 20 requests
- Headers include rate limit information

## WebSocket Support

Real-time updates available via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8001/ws');
ws.on('message', (data) => {
  const event = JSON.parse(data);
  console.log('Event:', event);
});
```

## Examples

See the `examples/` directory for complete API usage examples in various languages.