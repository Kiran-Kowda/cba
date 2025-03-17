import streamlit as st
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tenacity import retry, stop_after_attempt, wait_exponential
import time

def process_conversations(data):
    conversation_details = []
    
    for conversation in data['conversations']:
        # Format the date
        created_at = datetime.strptime(conversation['created_at'].split('+')[0], "%Y-%m-%dT%H:%M:%S.%f")
        formatted_date = created_at.strftime("%B %d, %Y %H:%M:%S")
        
        last_assistant_response = None
        last_assistant_score = "N/A"
        
        # First, find the last assistant response
        for message in conversation['messages']:
            if message.get('role') == 'assistant' and message.get('type') == 'text':
                last_assistant_response = message.get('content')
                last_assistant_score = message.get('score', "N/A")
        
        # Then collect all user questions
        for message in conversation['messages']:
            if message.get('role') == 'user':
                user_question = message.get('content')
                conversation_details.append({
                    'S.No': len(conversation_details) + 1,
                    'Asked at': formatted_date,  # Changed from 'Created At'
                    'Country': conversation['country'],
                    'User Question': user_question,
                    'Assistant Response': last_assistant_response,
                    'Score': last_assistant_score
                })
    
    return pd.DataFrame(conversation_details)

def get_daily_metrics(df):
    # Convert to datetime if not already
    df['Date'] = pd.to_datetime(df['Asked at']).dt.date  # Changed from 'Created At'
    
    # Get daily question count
    daily_questions = df.groupby('Date').agg({
        'User Question': 'count'  # Count questions per day
    }).reset_index()
    daily_questions.columns = ['Date', 'Questions']
    
    # Get daily unique users (countries)
    daily_users = df.groupby('Date').agg({
        'Country': 'nunique'  # Count unique countries per day
    }).reset_index()
    daily_users.columns = ['Date', 'Users']
    
    # Merge metrics
    daily_metrics = pd.merge(daily_questions, daily_users, on='Date')
    return daily_metrics

def get_country_metrics(df):
    # Get metrics by country
    country_metrics = df.groupby('Country').agg({
        'User Question': 'count',  # Count of questions
        'Asked at': 'nunique'      # Count of unique timestamps for users
    }).reset_index()
    
    country_metrics.columns = ['Country', 'Questions', 'Users']
    return country_metrics.sort_values('Questions', ascending=False)

# Add these functions after your existing functions
# Remove this unused function
# def setup_gemini():
#     GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
#     genai.configure(api_key=GOOGLE_API_KEY)
#     return genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_question_category(question):
    question = question.lower()
    
    # Check for Atlan branded questions
    if "atlan" in question:
        if any(word in question for word in ["what is", "what's", "define", "explain", "describe"]):
            return "What is / Define (Branded)"
        elif any(word in question for word in ["how to", "how do", "guide", "steps", "process"]):
            return "FAQ / How to (Branded)"
        return "Branded"  # fallback for other branded questions
    
    # Non-branded questions
    if any(word in question for word in ["what is", "what's", "define", "explain", "describe"]):
        return "What is / Define (Non branded)"
    elif any(word in question for word in ["how to", "how do", "guide", "steps", "process"]):
        return "FAQ / How to Type (Non branded)"
    
    return "Others"

def analyze_batch_questions(questions, batch_size=250):
    all_categories = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        batch_categories = [analyze_question_category(q) for q in batch]
        all_categories.extend(batch_categories)
        
        progress = min((i + batch_size) / len(questions), 1.0)
        yield progress, i + min(batch_size, len(questions) - i), all_categories

def get_category_metrics(df):
    questions = df['User Question'].tolist()
    total = len(questions)
    final_categories = None
    
    with st.spinner('Analyzing questions in batches... This may take a while.'):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for progress, processed, categories in analyze_batch_questions(questions):
            final_categories = categories
            progress_bar.progress(progress)
            status_text.write(f"Processed {processed} of {total} questions")
    
    df['Category'] = final_categories
    # Calculate counts and percentages
    category_metrics = df.groupby('Category')['User Question'].count().reset_index(name='Count')
    category_metrics['Percentage'] = (category_metrics['Count'] / category_metrics['Count'].sum() * 100).round(2)
    category_metrics['Percentage'] = category_metrics['Percentage'].astype(str) + '%'
    category_metrics = category_metrics.sort_values('Count', ascending=False)
    return category_metrics, df

def main():
    st.title("📊 Chatbase Conversations Analyzer")
    st.write("📤 Upload the Chatbase JSON export file to analyze user conversation details")
    
    uploaded_file = st.file_uploader("Choose a JSON file", type="json")
    
    # Fix the file upload check condition
    if (uploaded_file is not None and 
        ('previous_file' not in st.session_state or 
         st.session_state.get('previous_file') != getattr(uploaded_file, 'name', None))):
        st.session_state.category_analysis_done = False
        st.session_state.previous_file = uploaded_file.name
    
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            df = process_conversations(data)
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Full Data View", "📈 Daily Analytics", "🌎 Country Analytics", "🔍 Categorical Analysis"])
            
            with tab1:
                # Original content
                df['Asked at'] = pd.to_datetime(df['Asked at'])  # Changed from 'Created At'
                df = df.sort_values('Asked at', ascending=False)  # Changed from 'Created At'
                df['Asked at'] = df['Asked at'].dt.strftime("%B %d, %Y %H:%M:%S")  # Changed from 'Created At'
                df['S.No'] = range(1, len(df) + 1)
                
                st.dataframe(df)
                
                # Download button and statistics
                csv = df.to_csv(index=False)
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f'conversation_details_{current_time}.csv',
                    mime='text/csv',
                )
                
                st.subheader("⚡ Quick Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💬 Total Conversations", len(df))
                with col2:
                    st.metric("🌍 Unique Countries", len(df['Country'].unique()))
                with col3:
                    st.metric("⭐ Average Response Score", f"{df['Score'].mean():.2f}")
            
            with tab2:
                st.subheader("📊 Daily Usage Analytics")
                
                # Get daily metrics
                daily_metrics = get_daily_metrics(df)
                daily_metrics = daily_metrics.sort_values('Date')
                
                # Display metrics as a line chart
                st.line_chart(daily_metrics.set_index('Date'))
                
                # Display metrics as a table
                st.subheader("📅 Daily Breakdown")
                st.dataframe(daily_metrics.sort_values('Date', ascending=False))
                
                # Download daily metrics
                csv_metrics = daily_metrics.to_csv(index=False)
                st.download_button(
                    label="Download Daily Metrics",
                    data=csv_metrics,
                    file_name=f'daily_metrics_{current_time}.csv',
                    mime='text/csv',
                )
            
            with tab3:
                st.subheader("🌎 Country Usage Analytics")
                
                # Get country metrics without date filtering
                country_metrics = get_country_metrics(df)
                
                # Top countries metrics
                st.subheader("🏆 Top Countries Overview")
                top_metrics_col1, top_metrics_col2, top_metrics_col3 = st.columns(3)
                
                with top_metrics_col1:
                    top_country = country_metrics.iloc[0]
                    st.metric("Most Active Country", 
                             f"{top_country['Country']}", 
                             f"{top_country['Questions']} questions")
                
                with top_metrics_col2:
                    total_questions = country_metrics['Questions'].sum()
                    st.metric("Total Questions", 
                             f"{total_questions:,}", 
                             f"Across {len(country_metrics)} countries")
                
                with top_metrics_col3:
                    avg_questions = country_metrics['Questions'].mean()
                    st.metric("Average Questions per Country", 
                             f"{avg_questions:.1f}")
                
                # Detailed country breakdown
                st.subheader("🌍 Country Breakdown")
                st.dataframe(
                    country_metrics,
                    use_container_width=True
                )
                
                # Download option
                csv_metrics = country_metrics.to_csv(index=False)
                st.download_button(
                    label="📥 Download Country Metrics",
                    data=csv_metrics,
                    file_name=f'country_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                )
            
            with tab4:
                st.subheader("🔍 Question Category Analysis")
                
                # Fix the button and analysis logic
                if st.button("🔄 Analyze Question Categories"):
                    try:
                        # Remove the model parameter
                        category_metrics, df_with_categories = get_category_metrics(df)
                        
                        # Save results in session state
                        st.session_state.category_metrics = category_metrics
                        st.session_state.df_with_categories = df_with_categories
                        st.session_state.category_analysis_done = True
                        
                    except Exception as e:
                        st.error(f"Error in category analysis: {str(e)}")
                
                # Show results if analysis is done
                if st.session_state.get('category_analysis_done', False):
                    # Display category distribution
                    fig = px.pie(st.session_state.category_metrics, 
                               values='Count',  # Use Count for correct proportions
                               names='Category',
                               title='Question Category Distribution',
                               hole=0.4,
                               color_discrete_sequence=px.colors.qualitative.Set3)
                    
                    # Update traces with percentage values from the table
                    fig.update_traces(
                        textposition='outside',
                        textinfo='label+text',  # Show label and custom text
                        text=st.session_state.category_metrics['Percentage'],  # Use percentage from table
                        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{text}",
                        texttemplate='%{label}<br>%{text}'  # Show both label and percentage
                    )
                    
                    # Update layout
                    fig.update_layout(
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5
                        ),
                        margin=dict(t=60, b=100)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display metrics table
                    st.subheader("📊 Category Breakdown")
                    st.dataframe(st.session_state.category_metrics)
                    
                    # Display detailed question breakdown
                    st.subheader("📝 Question Details")
                    question_details = st.session_state.df_with_categories[['User Question', 'Category']].copy()
                    question_details.index = range(1, len(question_details) + 1)  # Add serial numbers
                    question_details.index.name = 'S.No'
                    st.dataframe(question_details.reset_index(), use_container_width=True)
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_categories = st.session_state.category_metrics.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Category Summary",
                            data=csv_categories,
                            file_name=f'category_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            mime='text/csv',
                        )
                    
                    with col2:
                        csv_details = question_details.to_csv()
                        st.download_button(
                            label="📥 Download Question Details",
                            data=csv_details,
                            file_name=f'question_details_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            mime='text/csv',
                        )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
