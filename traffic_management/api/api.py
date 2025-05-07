#!/usr/bin/env python
"""
API module for the traffic management system.
Provides RESTful endpoints for traffic status, prediction, signal control, and route optimization.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import uvicorn
from pydantic import BaseModel, Field
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.traffic_prediction import TrafficPredictionModel
from models.signal_control import SignalControlModel
from models.route_optimization import RouteOptimizationModel

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Urban Traffic Management API",
    description="API for traffic status, prediction, signal control, and route optimization",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme
security = HTTPBearer()

# Pydantic models for request/response validation
class TrafficStatusResponse(BaseModel):
    timestamp: float = Field(..., description="Current timestamp")
    status: Dict[str, Dict[str, Any]] = Field(..., description="Traffic status for each road segment")
    
class TrafficPredictionRequest(BaseModel):
    road_segments: List[str] = Field(..., description="List of road segment IDs to predict")
    prediction_horizon: int = Field(6, description="Prediction horizon in time steps (default: 6)")
    
class TrafficPredictionResponse(BaseModel):
    timestamp: float = Field(..., description="Current timestamp")
    predictions: Dict[str, List[Dict[str, Any]]] = Field(..., description="Predictions for each road segment")
    
class SignalControlRequest(BaseModel):
    intersection_id: str = Field(..., description="ID of the intersection")
    current_phase: int = Field(..., description="Current signal phase")
    queue_lengths: List[int] = Field(..., description="Queue lengths for each approach")
    waiting_times: List[float] = Field(..., description="Average waiting times for each approach")
    
class SignalControlResponse(BaseModel):
    intersection_id: str = Field(..., description="ID of the intersection")
    recommended_phase: int = Field(..., description="Recommended signal phase")
    phase_duration: int = Field(..., description="Recommended phase duration in seconds")
    
class RouteOptimizationRequest(BaseModel):
    origin: str = Field(..., description="Origin node ID")
    destination: str = Field(..., description="Destination node ID")
    departure_time: Optional[float] = Field(None, description="Departure time (timestamp)")
    max_alternatives: int = Field(3, description="Maximum number of alternative routes to return")
    
class RouteResponse(BaseModel):
    path: List[str] = Field(..., description="Sequence of node IDs in the route")
    segments: List[Dict[str, Any]] = Field(..., description="Details for each road segment in the route")
    total_time: float = Field(..., description="Estimated total travel time in seconds")
    total_distance: float = Field(..., description="Total distance in kilometers")
    avg_congestion: float = Field(..., description="Average congestion level (0-1)")
    
class RouteOptimizationResponse(BaseModel):
    timestamp: float = Field(..., description="Current timestamp")
    origin: str = Field(..., description="Origin node ID")
    destination: str = Field(..., description="Destination node ID")
    routes: List[RouteResponse] = Field(..., description="Recommended routes")

# Global variables for models
config_file = 'configs/default.json'
traffic_prediction_model = None
signal_control_model = None
route_optimization_model = None
last_model_update = 0
model_update_interval = 60  # seconds

# Authentication
def get_api_config():
    """Get API configuration from config file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config.get('api', {})
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def verify_token(token: str):
    """Verify JWT token."""
    api_config = get_api_config()
    auth_config = api_config.get('authentication', {})
    
    if not auth_config.get('enabled', False):
        return True
    
    try:
        secret_key = auth_config.get('secret_key', 'YOUR_SECRET_KEY')
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        return True
    except jwt.PyJWTError:
        return False

async def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Authenticate request using JWT."""
    api_config = get_api_config()
    auth_config = api_config.get('authentication', {})
    
    if not auth_config.get('enabled', False):
        return True
    
    token = credentials.credentials
    if not verify_token(token):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Middleware for rate limiting."""
    api_config = get_api_config()
    rate_limit_config = api_config.get('rate_limit', {})
    
    if not rate_limit_config.get('enabled', False):
        return await call_next(request)
    
    # In a real implementation, this would check a Redis cache or similar
    # for the number of requests from this client
    # For now, we'll just pass through
    
    response = await call_next(request)
    return response

# Model loading and updating
def load_models():
    """Load all models."""
    global traffic_prediction_model, signal_control_model, route_optimization_model, last_model_update
    
    try:
        traffic_prediction_model = TrafficPredictionModel(config_file)
        signal_control_model = SignalControlModel(config_file)
        route_optimization_model = RouteOptimizationModel(config_file)
        
        # Load map for route optimization
        map_file = 'data/maps/city_network.osm'
        if os.path.exists(map_file):
            route_optimization_model.load_map(map_file)
        
        last_model_update = time.time()
        logger.info("All models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def update_models_if_needed():
    """Update models if needed based on update interval."""
    global last_model_update
    
    current_time = time.time()
    if current_time - last_model_update > model_update_interval:
        # In a real implementation, this would check for updated model files
        # and reload them if necessary
        last_model_update = current_time

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "api": "AI-Powered Urban Traffic Management",
        "version": "0.1.0",
        "endpoints": [
            "/api/v1/traffic/status",
            "/api/v1/traffic/prediction",
            "/api/v1/signals/control",
            "/api/v1/routes/optimize"
        ]
    }

@app.get("/api/v1/traffic/status", response_model=TrafficStatusResponse)
async def get_traffic_status(authenticated: bool = Depends(authenticate)):
    """
    Get current traffic status for all monitored road segments.
    """
    update_models_if_needed()
    
    if route_optimization_model is None:
        if not load_models():
            raise HTTPException(status_code=500, detail="Failed to load models")
    
    try:
        # Get current traffic data from route optimization model
        # In a real implementation, this would be from a live data source
        
        # For demo, create mock data from the graph
        traffic_status = {}
        
        for edge in route_optimization_model.graph.edges():
            source, target = edge
            edge_id = f"{source}_{target}"
            
            edge_data = route_optimization_model.graph.get_edge_data(source, target)
            congestion = edge_data.get('congestion', 0.0)
            
            # Mock data based on graph properties
            traffic_status[edge_id] = {
                'source': source,
                'target': target,
                'congestion': congestion,
                'speed': edge_data.get('speed', edge_data.get('max_speed', 50.0)),
                'volume': int(congestion * 100),  # Mock volume based on congestion
                'last_updated': time.time()
            }
        
        return {
            'timestamp': time.time(),
            'status': traffic_status
        }
    except Exception as e:
        logger.error(f"Error getting traffic status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/traffic/prediction", response_model=TrafficPredictionResponse)
async def predict_traffic(request: TrafficPredictionRequest, authenticated: bool = Depends(authenticate)):
    """
    Predict traffic conditions for specified road segments.
    """
    update_models_if_needed()
    
    if traffic_prediction_model is None:
        if not load_models():
            raise HTTPException(status_code=500, detail="Failed to load models")
    
    try:
        # In a real implementation, this would use the trained LSTM model
        # For demo, generate mock predictions
        
        predictions = {}
        current_time = time.time()
        
        for segment in request.road_segments:
            # Parse segment ID to get source and target
            parts = segment.split('_')
            if len(parts) != 2:
                continue
                
            source, target = parts
            
            # Check if edge exists in graph
            if not route_optimization_model.graph.has_edge(source, target):
                continue
                
            edge_data = route_optimization_model.graph.get_edge_data(source, target)
            current_congestion = edge_data.get('congestion', 0.0)
            max_speed = edge_data.get('max_speed', 50.0)
            current_speed = edge_data.get('speed', max_speed)
            
            # Generate mock predictions
            segment_predictions = []
            for i in range(request.prediction_horizon):
                # Simple trend model - oscillating congestion
                predicted_congestion = current_congestion + 0.1 * (i % 3 - 1)
                predicted_congestion = max(0.0, min(1.0, predicted_congestion))
                
                # Speed decreases as congestion increases
                predicted_speed = max_speed * (1.0 - 0.7 * predicted_congestion)
                
                segment_predictions.append({
                    'timestamp': current_time + (i + 1) * 300,  # 5-minute intervals
                    'congestion': predicted_congestion,
                    'speed': predicted_speed,
                    'volume': int(predicted_congestion * 100),  # Mock volume
                })
            
            predictions[segment] = segment_predictions
        
        return {
            'timestamp': current_time,
            'predictions': predictions
        }
    except Exception as e:
        logger.error(f"Error predicting traffic: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/signals/control", response_model=SignalControlResponse)
async def control_signals(request: SignalControlRequest, authenticated: bool = Depends(authenticate)):
    """
    Get recommended signal control for an intersection.
    """
    update_models_if_needed()
    
    if signal_control_model is None:
        if not load_models():
            raise HTTPException(status_code=500, detail="Failed to load models")
    
    try:
        # In a real implementation, this would use the RL model
        # For demo, use a simple heuristic
        
        # Get queue lengths and waiting times
        queue_lengths = request.queue_lengths
        waiting_times = request.waiting_times
        
        # Simple heuristic: choose the phase with the highest combination of queue length and waiting time
        combined_score = [queue * wait for queue, wait in zip(queue_lengths, waiting_times)]
        
        # If current phase has a high score, keep it longer
        current_phase = request.current_phase
        current_phase_score = combined_score[current_phase] if current_phase < len(combined_score) else 0
        
        if current_phase_score > 0.8 * max(combined_score):
            recommended_phase = current_phase
        else:
            recommended_phase = combined_score.index(max(combined_score))
        
        # Adjust phase duration based on congestion
        base_duration = 30  # seconds
        phase_duration = base_duration
        
        if max(combined_score) > 0:
            # Adjust duration based on congestion
            congestion_factor = combined_score[recommended_phase] / max(combined_score)
            phase_duration = int(base_duration * (1 + congestion_factor))
            phase_duration = min(60, phase_duration)  # Cap at 60 seconds
        
        return {
            'intersection_id': request.intersection_id,
            'recommended_phase': recommended_phase,
            'phase_duration': phase_duration
        }
    except Exception as e:
        logger.error(f"Error controlling signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/routes/optimize", response_model=RouteOptimizationResponse)
async def optimize_route(request: RouteOptimizationRequest, authenticated: bool = Depends(authenticate)):
    """
    Get optimized routes between origin and destination.
    """
    update_models_if_needed()
    
    if route_optimization_model is None:
        if not load_models():
            raise HTTPException(status_code=500, detail="Failed to load models")
    
    try:
        # Use route optimization model to find best routes
        origin = request.origin
        destination = request.destination
        max_alternatives = min(request.max_alternatives, 5)  # Limit to 5 alternatives
        
        # Check if nodes exist
        if not route_optimization_model.graph.has_node(origin):
            raise HTTPException(status_code=400, detail=f"Origin node '{origin}' not found")
        if not route_optimization_model.graph.has_node(destination):
            raise HTTPException(status_code=400, detail=f"Destination node '{destination}' not found")
        
        # Find alternative paths
        data = {'origin': origin, 'destination': destination}
        routes = route_optimization_model.predict(data)
        
        # Limit to requested number of alternatives
        routes = routes[:max_alternatives]
        
        # Format the response
        formatted_routes = []
        for route in routes:
            formatted_routes.append(RouteResponse(
                path=route['path'],
                segments=route['segments'],
                total_time=route['total_time'],
                total_distance=route['total_distance'],
                avg_congestion=route['avg_congestion']
            ))
        
        return RouteOptimizationResponse(
            timestamp=time.time(),
            origin=origin,
            destination=destination,
            routes=formatted_routes
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing route: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
def update_traffic_data(background_tasks: BackgroundTasks):
    """Background task to update traffic data."""
    # In a real implementation, this would fetch and process real-time data
    background_tasks.add_task(_update_traffic_data_task)

async def _update_traffic_data_task():
    """Task to update traffic data."""
    if route_optimization_model is None:
        return
    
    try:
        # Mock traffic data update
        current_time = time.time()
        mock_traffic_data = {}
        
        for edge in route_optimization_model.graph.edges():
            source, target = edge
            
            # Generate random congestion changes
            edge_data = route_optimization_model.graph.get_edge_data(source, target)
            current_congestion = edge_data.get('congestion', 0.0)
            
            # Random walk for congestion
            congestion_change = (np.random.random() - 0.5) * 0.1
            new_congestion = max(0.0, min(1.0, current_congestion + congestion_change))
            
            mock_traffic_data[edge] = {
                'congestion': new_congestion,
                'volume': int(new_congestion * 100),
                'speed': edge_data.get('max_speed', 50.0) * (1.0 - 0.7 * new_congestion)
            }
        
        # Update route optimization model
        route_optimization_model.update_traffic_data(mock_traffic_data, current_time)
        
        logger.debug("Traffic data updated in background task")
    except Exception as e:
        logger.error(f"Error in background task: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Startup event to initialize models and logging."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/api.log')
        ]
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Load models
    load_models()
    
    logger.info("API started successfully")

# Run the API server
def start_api_server():
    """Start the API server."""
    api_config = get_api_config()
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    debug = api_config.get('debug', False)
    
    # Load models first
    load_models()
    
    # Run the server
    uvicorn.run("api.api:app", host=host, port=port, reload=debug)

if __name__ == "__main__":
    start_api_server() 