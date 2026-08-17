# app.py
import os
import certifi
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import Client
from langchain.agents import create_agent

os.environ['SSL_CERT_FILE'] = certifi.where()
load_dotenv()

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
WEATHERSTACK_API_KEY = os.getenv('WEATHERSTACK_API_KEY')

# Initialize tools
search_tool = TavilySearchResults(max_results=2)

@tool
def get_weather_data(city: str):
    """Fetch current weather information for a city"""
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )
    response = requests.get(url)
    data = response.json()
    
    if "current" not in data:
        return f"Could not fetch weather data for {city}"
    
    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%\n"
        f"Wind Speed: {data['current']['wind_speed']} km/h"
    )

# Function to initialize the agent
def initialize_agent():
    client = Client()
    pull_prompt = client.pull_prompt(
        "hwchase17/react", 
        include_model=True,
        dangerously_pull_public_prompt=True
    )
    prompt = pull_prompt.template
    tools = [search_tool, get_weather_data]
    
    agent = create_agent(
        model='deepseek-chat',
        tools=tools,
        system_prompt=prompt
    )
    return agent

# Streamlit UI
st.set_page_config(
    page_title="AI Agent Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Agent Assistant")
st.markdown("""
    This AI agent can help you with:
    - 🌐 **Web Search** - Find information from the internet
    - 🌤️ **Weather** - Get current weather for any city
    - 📚 **General Knowledge** - Answer questions using its knowledge
""")

# Sidebar for API key management
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    # API Key inputs (optional - if you want users to enter their own keys)
    use_env_keys = st.checkbox("Use environment variables", value=True)
    
    if not use_env_keys:
        deepseek_key = st.text_input("DeepSeek API Key", type="password")
        tavily_key = st.text_input("Tavily API Key", type="password")
        weather_key = st.text_input("WeatherStack API Key", type="password")
        
        # Update environment variables if provided
        if deepseek_key:
            os.environ['DEEPSEEK_API_KEY'] = deepseek_key
        if tavily_key:
            os.environ['TAVILY_API_KEY'] = tavily_key
        if weather_key:
            os.environ['WEATHERSTACK_API_KEY'] = weather_key
    
    st.markdown("---")
    st.markdown("### 📝 Tips")
    st.markdown("""
    - Try asking: "What's the weather in London?"
    - Try searching: "Who is the CEO of Tesla?"
    - Ask general questions: "What is quantum computing?"
    """)
    
    # Show current API status
    st.markdown("---")
    st.markdown("### 🔑 API Status")
    st.success("✅ DeepSeek: Connected") if os.getenv('DEEPSEEK_API_KEY') else st.error("❌ DeepSeek: Missing")
    st.success("✅ Tavily: Connected") if os.getenv('TAVILY_API_KEY') else st.error("❌ Tavily: Missing")
    st.success("✅ WeatherStack: Connected") if os.getenv('WEATHERSTACK_API_KEY') else st.error("❌ WeatherStack: Missing")

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "agent" not in st.session_state:
    with st.spinner("Initializing AI Agent..."):
        try:
            st.session_state.agent = initialize_agent()
            st.success("Agent initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the agent
                response = st.session_state.agent.invoke({
                    "messages": [{"role": "user", "content": prompt}]
                })
                
                # Extract and display the response
                assistant_response = response["messages"][-1].content
                st.markdown(assistant_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

# Clear chat button in sidebar
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        Powered by DeepSeek, Tavily, and WeatherStack APIs
    </div>
""", unsafe_allow_html=True)