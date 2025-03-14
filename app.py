import streamlit as st
import os
import json
import time
import random
from typing import Dict, List, Any, Tuple, Generator
import anthropic
from datetime import datetime
from langgraph.graph import StateGraph, END
from typing import TypedDict
import matplotlib.pyplot as plt
from components.agent_visualizer import visualize_agent_workflow
import html
import asyncio

# Import our agent functions
from agents.coordinator import coordinator_agent
from agents.route_optimizer import route_optimizer_agent
from agents.fleet_monitor import fleet_monitor_agent
from agents.data_retriever import data_retriever_agent
from agents.notification import notification_agent
from utils.api_mock import load_mock_data

# Initialize Claude client (using environment variable for API key)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "dummy_key"))

# Define the state schema
class AgentState(TypedDict):
    input: str
    context: Dict[str, Any]
    next: str

# Set page configuration (removing the unsupported 'theme' parameter)
st.set_page_config(
    page_title="Supply Chain Multi-Agent System",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Keep the CSS styling to make the app dark themed with proper text contrast
st.markdown("""
<style>
    /* Dark theme background */
    .stApp {
        background-color: #262730;
    }
    
    /* Text elements */
    h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stText {
        color: auto !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        color: black !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
        font-weight: bold;
        border: 1px solid gray;
    }
    
    /* Custom chat styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-message.user {
        background-color: #4c8bf5;
        color: white;
    }
    .chat-message.assistant {
        background-color: #343a40;
        color: white;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1oe6o3n {
        background-color: #1e2129;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #4c8bf5 !important;
    }
    .stAlert > div {
        color: white !important;
    }
    
    /* Agent thinking status */
    .agent-thinking {
        padding: 1rem;
        background-color: #3c4043;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #ffd166;
    }
    
    /* Status message container */
    .chat-message.status {
        background-color: #2b3035;
        border-left: 5px solid #ffd166;
        color: #e0e0e0;
        padding: 0.8rem;
        font-size: 0.9em;
        margin-bottom: 0.5rem;
    }
    
    /* Fix for agent header */
    .agent-header {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .agent-icon {
        background-color: #ffd166;
        color: #333;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 8px;
        font-weight: bold;
        text-align: center;
    }
    
    .agent-name {
        font-weight: bold;
        color: #ffd166;
    }
    
    /* Status steps */
    .status-step {
        margin: 5px 0;
        display: flex;
        align-items: center;
    }
    
    .status-icon {
        margin-right: 8px;
        color: #00AA66;
        display: inline-block;
    }
    
    .status-icon.processing {
        color: #ffd166;
    }
    
    .status-text {
        flex: 1;
    }
    
    /* Pending step styling */
    .status-step.pending {
        opacity: 0.7;
    }
    
    .status-icon.pending {
        color: #8c8c8c;
    }
</style>
""", unsafe_allow_html=True)

def build_agent_network():
    """Build the network of agents using LangGraph."""
    # Create the graph with defined state type
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("route_optimizer", route_optimizer_agent)
    workflow.add_node("fleet_monitor", fleet_monitor_agent)
    workflow.add_node("data_retriever", data_retriever_agent)
    workflow.add_node("notification", notification_agent)
    
    # Define the routing function
    def router(state: AgentState) -> str:
        return state["next"]
    
    # Add conditional edges based on the router function
    workflow.add_conditional_edges(
        "coordinator",
        router,
        {
            "route_optimizer": "route_optimizer",
            "fleet_monitor": "fleet_monitor",
            "data_retriever": "data_retriever",
            "notification": "notification"
        }
    )
    
    # Add conditional edges for route_optimizer
    workflow.add_conditional_edges(
        "route_optimizer",
        router,
        {
            "data_retriever": "data_retriever",
            "notification": "notification",
            END: END
        }
    )
    
    # Add conditional edges for fleet_monitor
    workflow.add_conditional_edges(
        "fleet_monitor",
        router,
        {
            "data_retriever": "data_retriever",
            "notification": "notification",
            END: END
        }
    )
    
    # Add conditional edges for data_retriever and notification
    workflow.add_conditional_edges(
        "data_retriever",
        router,
        {END: END}
    )
    
    workflow.add_conditional_edges(
        "notification",
        router,
        {END: END}
    )
    
    # Set the entry point
    workflow.set_entry_point("coordinator")
    
    # Compile the graph
    return workflow.compile()

def get_demo_scenarios(role: str) -> Dict[str, str]:
    """Provide demo scenarios based on user role."""
    if role == "logistics_coordinator":
        return {
            "Optimize Route": "I need to optimize delivery routes for tomorrow. We have 12 deliveries to make across the city. What's the most efficient route?",
            "Check Weather Impact": "How will the forecasted rain affect our delivery schedule tomorrow?",
            "Urgent Delivery": "A high-priority customer needs a package delivered ASAP. Can we reroute a nearby driver?",
            "Delivery Exception": "Order #45982 was rejected by the customer. What should the driver do now?"
        }
    elif role == "driver":
        return {
            "Navigation Help": "I'm at Main St and 5th Ave. What's my best route to the next stop considering current traffic?",
            "Delivery Problem": "I can't access the delivery location. The gate is locked. What should I do?",
            "Vehicle Issue": "My truck is showing a check engine light. Should I continue my route?",
            "Schedule Update": "How many more deliveries do I have today and what's my estimated completion time?"
        }
    else:  # admin
        return {
            "Fleet Overview": "Show me a summary of our entire fleet status right now.",
            "Maintenance Schedule": "Which vehicles are due for maintenance in the next week?",
            "Driver Performance": "Who are our top-performing drivers this month based on on-time delivery rate?",
            "Fuel Efficiency": "Give me a report on fuel consumption across our fleet for the past month."
        }

def get_agent_details(agent_name):
    """Get detailed information about a specific agent."""
    agent_details = {
        "coordinator": {
            "name": "Coordinator",
            "description": "Central routing agent that determines which specialized agent to call",
            "capabilities": ["Request analysis", "Intent recognition", "Agent routing"],
            "status": ["Analyzing request...", "Determining request type...", "Selecting specialized agent..."]
        },
        "route_optimizer": {
            "name": "Route Optimizer",
            "description": "Specializes in finding optimal delivery routes",
            "capabilities": ["Path optimization", "ETA calculation", "Traffic incorporation"],
            "status": ["Calculating optimal routes...", "Analyzing traffic patterns...", "Optimizing for fuel efficiency...", "Generating turn-by-turn directions..."]
        },
        "fleet_monitor": {
            "name": "Fleet Monitor",
            "description": "Tracks vehicle status, maintenance, and driver performance",
            "capabilities": ["Vehicle tracking", "Maintenance scheduling", "Performance analysis"],
            "status": ["Retrieving vehicle locations...", "Checking maintenance records...", "Analyzing driver performance..."]
        },
        "data_retriever": {
            "name": "Data Retriever",
            "description": "Gets external data like weather and traffic conditions",
            "capabilities": ["Weather data", "Traffic information", "Road conditions"],
            "status": ["Fetching current weather data...", "Retrieving traffic information...", "Checking road conditions..."]
        },
        "notification": {
            "name": "Notification Service",
            "description": "Sends alerts and updates to drivers and stakeholders",
            "capabilities": ["SMS alerts", "Email updates", "In-app notifications"],
            "status": ["Preparing notification content...", "Determining recipients...", "Sending notifications..."]
        }
    }
    return agent_details.get(agent_name, {"name": agent_name, "description": "Unknown agent", "capabilities": [], "status": []})

def simulate_agent_progress(agent_name, step_container):
    """Simulate the progress of an agent with detailed status updates."""
    agent = get_agent_details(agent_name)
    
    # Display agent info header with more prominent styling
    step_container.markdown(f"""
    <div style='background-color: #3c4043; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid #ffd166;'>
        <h3 style='margin:0; color: white;'>{agent['name']} Agent</h3>
        <p style='margin:8px 0 0 0; color: #cccccc; font-size: 0.95em;'>{agent['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status list with larger, more visible container
    status_container = step_container.empty()
    
    # Simulate steps with typing effect
    statuses = []
    for i, status in enumerate(agent["status"]):
        statuses.append(status)
        status_html = ""
        for s in statuses:
            if s == status:
                status_html += f"<li style='color: #ffd166; font-size: 1.1em; margin-bottom: 12px;'>⟳ {s}</li>"
            else:
                status_html += f"<li style='color: #00AA66; font-size: 1.1em; margin-bottom: 12px;'>✓ {s}</li>"
        
        status_container.markdown(f"""
        <div style='background-color: #262730; padding: 12px 20px; border-radius: 8px;'>
            <ul style='list-style-type: none; padding-left: 5px; margin-bottom: 0;'>
                {status_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a small delay to simulate processing
        time.sleep(0.8)
    
    # Mark all as complete
    status_html = ""
    for s in statuses:
        status_html += f"<li style='color: #00AA66; font-size: 1.1em; margin-bottom: 12px;'>✓ {s}</li>"
    
    status_container.markdown(f"""
    <div style='background-color: #262730; padding: 12px 20px; border-radius: 8px;'>
        <ul style='list-style-type: none; padding-left: 5px; margin-bottom: 0;'>
            {status_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Add completion message
    step_container.markdown(f"""
    <div style='text-align: right; color: #00AA66; font-weight: bold; margin-top: 12px; font-size: 1.1em;'>
        Task completed successfully ✓
    </div>
    """, unsafe_allow_html=True)

def update_with_thinking(msg, agent):
    """Update the thinking status with more detailed information."""
    agent_details = get_agent_details(agent)
    
    thinking_html = f"""
    <div style='margin-bottom: 10px;'>
        <h4 style='margin:0; color: white;'>{agent_details['name']} Agent Processing</h4>
        <p style='color: #cccccc; margin-top: 5px;'>{agent_details['description']}</p>
        <div style='margin-top: 10px;'>
            <p style='color: white;'><strong>Capabilities:</strong></p>
            <ul style='color: #cccccc;'>
    """
    for capability in agent_details['capabilities']:
        thinking_html += f"<li>{capability}</li>"
    
    thinking_html += """
            </ul>
        </div>
        <div style='margin-top: 10px;'>
            <p style='color: white;'><strong>Current Status:</strong></p>
            <p style='color: #ffd166;'>⟳ Processing request...</p>
        </div>
    </div>
    """
    return thinking_html

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'context' not in st.session_state:
        st.session_state.context = {
            "role": None,
            "mock_data": load_mock_data(),
            "conversation_history": [],
            "messages": []
        }
    print(st.session_state.context)
    if 'agent_network' not in st.session_state:
        st.session_state.agent_network = build_agent_network()

# Create a function for adding status messages to the chat
def add_status_message(agent_name, status_type="info", content=""):
    """Add agent status messages that display correctly."""
    agent_details = get_agent_details(agent_name)
    
    # Create a clean HTML string without any potential issues
    if status_type == "start":
        status_html = f"""
        <div class="agent-header">
            <span class="agent-icon">{agent_name[0].upper()}</span>
            <span class="agent-name">{agent_details['name']} Agent</span>
        </div>
        <p>{agent_details['description']}</p>
        <div class="status-step">
            <span class="status-icon processing">⟳</span>
            <span class="status-text">Processing your request...</span>
        </div>
        """
    elif status_type == "step":
        status_html = f"""
        <div class="status-step">
            <span class="status-icon processing">⟳</span>
            <span class="status-text">{content}</span>
        </div>
        """
    elif status_type == "complete":
        status_html = f"""
        <div class="status-step">
            <span class="status-icon">✓</span>
            <span class="status-text">Processing complete</span>
        </div>
        """
    
    # Add the status message to the conversation
    st.session_state.messages.append({
        "role": "status", 
        "agent": agent_name, 
        "content": status_html.strip()
    })

# Add this function to determine next agent based on user message
def determine_next_agent(user_message):
    """Determine which specialized agent to call based on the message content."""
    user_message = user_message.lower()
    if "route" in user_message or "path" in user_message or "delivery" in user_message:
        return "route_optimizer"
    elif "fleet" in user_message or "vehicle" in user_message or "driver" in user_message:
        return "fleet_monitor"
    elif "weather" in user_message or "traffic" in user_message or "condition" in user_message:
        return "data_retriever"
    elif "notify" in user_message or "alert" in user_message or "send" in user_message:
        return "notification"
    return None

# Completely new approach to process_user_message
def process_user_message(user_message: str, agent_status_placeholder=None):
    """Process user message with detailed animated status messages."""
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    # Add to context
    st.session_state.context["conversation_history"].append({"role": "user", "content": user_message})
    
    # Create a status placeholder in the UI
    status_placeholder = st.empty()
    
    # Determine which specialized agent to call
    specialized_agent = determine_next_agent(user_message)
    
    # Show initial processing status - COORDINATOR AGENT
    with status_placeholder:
        coordinator_html = f"""
        <div style="border-left: 5px solid #FF9500; padding: 15px; background-color: #2b3035; border-radius: 4px; margin-bottom: 15px;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="background-color: #FF9500; color: #333; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px; font-weight: bold;">C</div>
                <div style="font-weight: bold; color: #FF9500; font-size: 1.1em;">Coordinator Agent</div>
            </div>
            <div style="margin-bottom: 10px; color: #e0e0e0;">Central routing agent that determines which specialized agent to call</div>
            <div style="margin-left: 10px;">
                <div style="margin-bottom: 12px;">
                    <span style="color: #ffd166; margin-right: 8px;">⟳</span>
                    <span style="color: #ffd166; font-weight: bold;">Analyzing request...</span>
                </div>
                <div style="margin-bottom: 12px; opacity: 0.6;">
                    <span style="color: #8c8c8c; margin-right: 8px;">○</span>
                    <span style="color: #8c8c8c;">Determining request type...</span>
                </div>
                <div style="margin-bottom: 12px; opacity: 0.6;">
                    <span style="color: #8c8c8c; margin-right: 8px;">○</span>
                    <span style="color: #8c8c8c;">Selecting specialized agent...</span>
                </div>
            </div>
        </div>
        """
        st.markdown(coordinator_html, unsafe_allow_html=True)
    
    # Sleep to simulate step 1 completing
    time.sleep(1.2)
    
    # Update to show step 2 active
    with status_placeholder:
        coordinator_html = f"""
        <div style="border-left: 5px solid #FF9500; padding: 15px; background-color: #2b3035; border-radius: 4px; margin-bottom: 15px;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="background-color: #FF9500; color: #333; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px; font-weight: bold;">C</div>
                <div style="font-weight: bold; color: #FF9500; font-size: 1.1em;">Coordinator Agent</div>
            </div>
            <div style="margin-bottom: 10px; color: #e0e0e0;">Central routing agent that determines which specialized agent to call</div>
            <div style="margin-left: 10px;">
                <div style="margin-bottom: 12px;">
                    <span style="color: #00AA66; margin-right: 8px;">✓</span>
                    <span>Analyzing request...</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <span style="color: #ffd166; margin-right: 8px;">⟳</span>
                    <span style="color: #ffd166; font-weight: bold;">Determining request type...</span>
                </div>
                <div style="margin-bottom: 12px; opacity: 0.6;">
                    <span style="color: #8c8c8c; margin-right: 8px;">○</span>
                    <span style="color: #8c8c8c;">Selecting specialized agent...</span>
                </div>
            </div>
        </div>
        """
        st.markdown(coordinator_html, unsafe_allow_html=True)
        
    # Sleep to simulate step 2 completing
    time.sleep(1.2)
    
    # Update to show step 3 active
    with status_placeholder:
        coordinator_html = f"""
        <div style="border-left: 5px solid #FF9500; padding: 15px; background-color: #2b3035; border-radius: 4px; margin-bottom: 15px;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="background-color: #FF9500; color: #333; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px; font-weight: bold;">C</div>
                <div style="font-weight: bold; color: #FF9500; font-size: 1.1em;">Coordinator Agent</div>
            </div>
            <div style="margin-bottom: 10px; color: #e0e0e0;">Central routing agent that determines which specialized agent to call</div>
            <div style="margin-left: 10px;">
                <div style="margin-bottom: 12px;">
                    <span style="color: #00AA66; margin-right: 8px;">✓</span>
                    <span>Analyzing request...</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <span style="color: #00AA66; margin-right: 8px;">✓</span>
                    <span>Determining request type...</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <span style="color: #ffd166; margin-right: 8px;">⟳</span>
                    <span style="color: #ffd166; font-weight: bold;">Selecting specialized agent...</span>
                </div>
            </div>
        </div>
        """
        st.markdown(coordinator_html, unsafe_allow_html=True)
    
    # Sleep to simulate step 3 completing
    time.sleep(1.2)
    
    # If we've determined a specialized agent is needed, show that agent's status
    if specialized_agent:
        agent_details = get_agent_details(specialized_agent)
        agent_color = "#00AA66"  # Green for specialized agents
        agent_initials = specialized_agent[0].upper()
        
        # First, clear the placeholder to remove the coordinator
        status_placeholder.empty()
        
        # Create a new container for both agents
        with status_placeholder.container():
            # Coordinator agent (completed) - use a simpler format
            st.markdown(f"""
            <div style="border-left: 5px solid #FF9500; padding: 15px; background-color: #2b3035; border-radius: 4px; margin-bottom: 15px;">
                <div style="display: flex; align-items: center;">
                    <div style="background-color: #FF9500; color: #333; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px; font-weight: bold; text-align: center;">C</div>
                    <div style="font-weight: bold; color: #FF9500;">Coordinator Agent</div>
                </div>
                <div style="margin-top: 10px; color: #e0e0e0;">Central routing agent that determines which specialized agent to call</div>
                <ul style="list-style-type: none; padding-left: 10px; margin-top: 10px;">
                    <li style="margin-bottom: 8px;"><span style="color: #00AA66; margin-right: 8px;">✓</span> Analyzing request...</li>
                    <li style="margin-bottom: 8px;"><span style="color: #00AA66; margin-right: 8px;">✓</span> Determining request type...</li>
                    <li style="margin-bottom: 8px;"><span style="color: #00AA66; margin-right: 8px;">✓</span> Selecting specialized agent...</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Use Streamlit's native components for specialized agent
            st.markdown(f"""
            <div style="border-left: 5px solid {agent_color}; padding: 15px; background-color: #2b3035; border-radius: 4px; margin-bottom: 15px;">
                <div style="display: flex; align-items: center;">
                    <div style="background-color: {agent_color}; color: #333; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px; font-weight: bold; text-align: center;">{agent_initials}</div>
                    <div style="font-weight: bold; color: {agent_color};">{agent_details['name']} Agent</div>
                </div>
                <div style="margin-top: 10px; color: #e0e0e0;">{agent_details['description']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add status step container without the complex HTML
            status_step_container = st.container()
        
        # Process each step one by one using simpler components
        for step_index in range(len(agent_details["status"])):
            # Get the current step text
            current_step = agent_details["status"][step_index]
            
            # Clear previous content
            status_step_container.empty()
            
            # Display only the current step with the processing indicator
            with status_step_container:
                # Show current active step only
                st.markdown(f"""
                <div style='margin-bottom: 10px;'>
                    <span style='color: #ffd166; margin-right: 8px;'>⟳</span> 
                    <span style='color: #ffd166; font-weight: bold;'>{current_step}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Sleep before moving to next step
            time.sleep(1.2)
            
            # After sleep, mark this step as completed before moving to next one
            status_step_container.empty()
            with status_step_container:
                # Add the completed step to the list
                st.markdown(f"""
                <div style='margin-bottom: 10px;'>
                    <span style='color: #00AA66; margin-right: 8px;'>✓</span> {current_step}
                </div>
                """, unsafe_allow_html=True)
    
    # Process with LangGraph (actual processing happens here)
    try:
        # Process with LangGraph
        result = st.session_state.agent_network.invoke({
            "input": user_message,
            "context": st.session_state.context,
            "next": ""
        })
        
        # Clear the status placeholder after processing
        status_placeholder.empty()
        
        # Get response
        final_context = result.get("context", st.session_state.context)
        response = "I'm sorry, there was an error processing your request."
        if "messages" in final_context and final_context["messages"]:
            response = final_context["messages"][-1]["content"]
        
        # Add response to chat
        st.session_state.context = final_context
        st.session_state.context["messages"] = []
        st.session_state.context["conversation_history"].append({"role": "assistant", "content": response})
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Add agent tag to show which agent was used
        if specialized_agent:
            agent_details = get_agent_details(specialized_agent)
            agent_tag = f"""
            <div style="display: inline-block; margin-top: 5px; padding: 3px 8px; background-color: #00AA66; color: white; border-radius: 12px; font-size: 0.8em;">
                Processed by {agent_details['name']} Agent
            </div>
            """
            st.session_state.messages.append({"role": "agent_tag", "content": agent_tag})
        
    except Exception as e:
        # Clear the status placeholder
        status_placeholder.empty()
        
        # Handle errors with fallback
        try:
            fallback_response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=2000,
                system="You are a helpful supply chain assistant.",
                messages=[{"role": "user", "content": user_message}]
            )
            response = fallback_response.content[0].text
        except:
            response = f"I encountered an error: {str(e)}. Please try asking in a different way."
        
        st.session_state.context["conversation_history"].append({"role": "assistant", "content": response})
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    """Main function to run the Streamlit app."""
    initialize_session_state()
    
    # Page header
    st.title("🚚 Supply Chain Multi-Agent System")
    st.markdown("*Powered by Aisera*", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Role selection
        role_options = ["logistics_coordinator", "driver", "admin"]
        display_names = ["Logistics Coordinator", "Driver", "Administrator"]
        
        # If role isn't set yet, select it
        if not st.session_state.context["role"]:
            role_idx = st.selectbox(
                "Select your role:", 
                range(len(role_options)), 
                format_func=lambda i: display_names[i],
                key="role_select"
            )
            
            if st.button("Set Role", key="set_role_btn"):
                st.session_state.context["role"] = role_options[role_idx]
                
                # Add welcome message
                welcome_messages = {
                    "logistics_coordinator": "Welcome to the AI Route Optimization Assistant. How can I help you with your delivery coordination today?",
                    "driver": "Welcome to the Driver Assistant. I can provide route information, navigation, and updates for your deliveries today.",
                    "admin": "Welcome, Administrator. How may I assist you with fleet management and system monitoring today?"
                }
                
                system_msg = welcome_messages[st.session_state.context["role"]]
                st.session_state.messages.append({"role": "assistant", "content": system_msg})
                
                # Force a rerun to show the updated interface
                st.rerun()
        else:
            # Display current role
            st.info(f"Current Role: {display_names[role_options.index(st.session_state.context['role'])]}")
            
            # Provide an option to change roles
            if st.button("Change Role", key="change_role_btn"):
                st.session_state.context["role"] = None
                st.session_state.messages = []
                st.rerun()
            
            # Demo scenarios based on role
            st.header("Demo Scenarios")
            scenarios = get_demo_scenarios(st.session_state.context["role"])
            
            # Create demo buttons
            for scenario_name, scenario_text in scenarios.items():
                if st.button(scenario_name, key=f"scenario_{scenario_name.lower().replace(' ', '_')}"):
                    # Add user message to chat first
                    st.session_state.messages.append({"role": "user", "content": scenario_text})
                    
                    # Create a status placeholder
                    status_placeholder = st.empty()
                    
                    # Process the message with the placeholder
                    process_user_message(scenario_text)
                    
                    # Force rerun to show all messages and clear the placeholder
                    st.rerun()
        
        # Add API section
        st.header("API Access")
        st.markdown("""
        This multi-agent system is also available via API. Here's how to call it:
        """)
        
        if st.session_state.context["role"]:
            if st.session_state.messages and len(st.session_state.messages) > 0:
                # Get the last user message
                last_user_message = None
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "user":
                        last_user_message = msg["content"]
                        break
                
                if last_user_message:
                    curl_command = f"""
                    ```bash
                    curl -X POST http://localhost:8000/query \\
                      -H "Content-Type: application/json" \\
                      -d '{{"input": "{last_user_message}", "role": "{st.session_state.context["role"]}"}}' 
                    ```
                    """
                    st.markdown("#### Example API Call for Your Last Query:")
                    st.markdown(curl_command)
                    
                    st.markdown("#### Response Format:")
                    st.code("""
                    {
                      "output": {
                        "content": "The answer to your query...",
                        "response_metadata": {
                          "token_usage": {
                            "completion_tokens": 123,
                            "prompt_tokens": 456,
                            "total_tokens": 579
                          },
                          "model_name": "claude-3-sonnet-20240229"
                        },
                        "agent_used": "fleet_monitor"
                      }
                    }
                    """, language="json")
                    
                    # Add a direct link to run the API
                    st.markdown("#### API Documentation")
                    st.markdown("Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.")
    
    # Main chat area
    if st.session_state.context["role"]:
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user">
                        <div><strong>You</strong></div>
                        <div class="content">{message['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif message["role"] == "assistant":
                    st.markdown(f"""
                    <div class="chat-message assistant">
                        <div><strong>Assistant</strong></div>
                        <div class="content">{message['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif message["role"] == "status":
                    # Just render the HTML content directly
                    st.markdown(message['content'], unsafe_allow_html=True)
                elif message["role"] == "agent_tag":
                    st.markdown(message['content'], unsafe_allow_html=True)
        
        # Message input
        user_input = st.chat_input("Type your message here...")
        if user_input:
            process_user_message(user_input)
            st.rerun()
    
if __name__ == "__main__":
    main() 