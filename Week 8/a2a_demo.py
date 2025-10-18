import json
import time
import requests
import sqlite3
from typing import Dict, Any, List
from database import Database
import config


class Agent:
    """
    Base Agent class for A2A protocol.
    Each agent is autonomous and can query database directly.
    """
    
    def __init__(self, name: str, role: str, expertise: List[str], database: Database):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.database = database
        self.api_key = config.OPENAI_API_KEY
        self.api_url = config.OPENAI_API_URL
        self.model = config.OPENAI_MODEL
        self.db_path = config.DATABASE_PATH
        self.conversation_history = []
    
    def _get_db_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def query_database(self, query_type: str) -> Any:
        """Query database for information - agents have direct access to data."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        if query_type == "products":
            cursor.execute("SELECT name, type, users, description FROM products")
            rows = cursor.fetchall()
            result = [{"name": r[0], "type": r[1], "users": r[2], "description": r[3]} for r in rows]
        
        elif query_type == "company":
            cursor.execute("SELECT name, founded, employees, focus FROM company_info LIMIT 1")
            row = cursor.fetchone()
            result = {"name": row[0], "founded": row[1], "employees": row[2], "focus": row[3]} if row else {}
        
        elif query_type == "financial":
            cursor.execute("SELECT year, revenue, growth, funding FROM financial_data ORDER BY year DESC")
            rows = cursor.fetchall()
            result = [{"year": r[0], "revenue": r[1], "growth": r[2], "funding": r[3]} for r in rows]
        
        elif query_type == "employees":
            cursor.execute("SELECT name, role, department, expertise FROM employees")
            rows = cursor.fetchall()
            result = [{"name": r[0], "role": r[1], "department": r[2], "expertise": r[3]} for r in rows]
        
        else:
            result = None
        
        conn.close()
        return result
    
    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI API to generate agent's response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": config.MAX_TOKENS
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=config.TIMEOUT_SECONDS
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"
    
    def send_message(self, receiver: 'Agent', message: str) -> str:
        """
        Send message to another agent (A2A communication).
        This is peer-to-peer - agent talks directly to another agent.
        """
        start_time = time.time()
        
        print(f"\n[{self.name}] → [{receiver.name}]: {message}")
        
        response = receiver.receive_message(self, message)
        
        end_time = time.time()
        latency = end_time - start_time
        
        self.database.log_a2a_conversation(
            sender=self.name,
            receiver=receiver.name,
            message=message,
            response=response,
            metadata={"sender_role": self.role, "receiver_role": receiver.role, "latency": latency}
        )
        
        self.database.log_metric(
            protocol_type="A2A",
            metric_name="message_latency",
            metric_value=latency
        )
        
        print(f"Response: {response}")
        print(f"({latency:.2f}s)")
        
        return response
    
    def receive_message(self, sender: 'Agent', message: str) -> str:
        """
        Receive and process message from another agent.
        Agent can query database directly to get information for response.
        """
        db_context = {}
        msg_lower = message.lower()
        
        print(f"[{self.name}] Querying database...")
        
        if "product" in msg_lower or "sản phẩm" in msg_lower:
            db_context["products"] = self.query_database("products")
        
        if "financial" in msg_lower or "tài chính" in msg_lower or "revenue" in msg_lower:
            db_context["financial"] = self.query_database("financial")
        
        if "company" in msg_lower or "công ty" in msg_lower:
            db_context["company"] = self.query_database("company")
        
        if "team" in msg_lower or "employee" in msg_lower or "nhân viên" in msg_lower:
            db_context["employees"] = self.query_database("employees")
        
        system_message = f"""You are {self.name}, a {self.role}.
Your expertise: {', '.join(self.expertise)}

Database info:
{json.dumps(db_context, indent=2, ensure_ascii=False)}

Respond to {sender.name} professionally. Keep it concise (2-3 sentences)."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{sender.name} says: {message}"}
        ]
        
        response = self.call_llm(messages)
        return response
    
    def negotiate(self, other_agent: 'Agent', topic: str) -> Dict[str, str]:
        """
        Negotiate with another agent on a specific topic.
        Demonstrates collaborative A2A communication.
        """
        print(f"\n{'='*60}")
        print(f"Negotiation: {self.name} ↔ {other_agent.name}")
        print(f"Topic: {topic}")
        print(f"{'='*60}")
        
        proposal = f"Regarding {topic}, what is your opinion from a {other_agent.role} perspective?"
        response1 = self.send_message(other_agent, proposal)
        time.sleep(1)
        
        follow_up = f"Based on your input, I think we should consider both {self.role} and {other_agent.role} aspects. Do you agree?"
        response2 = other_agent.send_message(self, follow_up)
        
        return {
            "topic": topic,
            "agent1": self.name,
            "agent2": other_agent.name,
            "initial_response": response1,
            "follow_up_response": response2
        }


class ResearchAgent(Agent):
    """Research specialist agent."""
    
    def __init__(self, database: Database):
        super().__init__(
            name="Dr. Research",
            role="Research Scientist",
            expertise=["Machine Learning", "Data Analysis", "Academic Research"],
            database=database
        )


class BusinessAgent(Agent):
    """Business strategy agent."""
    
    def __init__(self, database: Database):
        super().__init__(
            name="Ms. Business",
            role="Business Strategist",
            expertise=["Market Analysis", "Business Development", "ROI Optimization"],
            database=database
        )


class EngineeringAgent(Agent):
    """Engineering implementation agent."""
    
    def __init__(self, database: Database):
        super().__init__(
            name="Mr. Engineer",
            role="Lead Engineer",
            expertise=["Software Architecture", "System Design", "Technical Implementation"],
            database=database
        )


def demo_a2a():
    """Run A2A protocol demo."""
    print("\n" + "="*60)
    print("A2A (Agent-to-Agent)")
    print("Peer-to-Peer: Agents query DB directly & communicate")
    print("="*60)
    
    db = Database()
    research_agent = ResearchAgent(db)
    business_agent = BusinessAgent(db)
    engineering_agent = EngineeringAgent(db)
    
    # Demo questions - SAME as MCP for comparison
    print("\n--- Question 1 ---")
    print("Q: Cho tôi biết thông tin về công ty và các sản phẩm chính?")
    business_agent.send_message(
        research_agent,
        "Can you tell me about the company info and main products from your research perspective?"
    )
    time.sleep(1)
    
    print("\n--- Question 2 ---")
    print("Q: Tình hình tài chính của công ty năm 2024 như thế nào?")
    research_agent.send_message(
        business_agent,
        "What's the company's financial situation in 2024? I need to understand the business metrics."
    )
    time.sleep(1)
    
    print("\n--- Question 3 ---")
    print("Q: Công ty có bao nhiêu nhân viên và họ làm việc ở các phòng ban nào?")
    business_agent.send_message(
        engineering_agent,
        "How many employees do we have and what departments are they in? Need to understand the team structure."
    )
    time.sleep(1)
    
    return {
        "protocol": "A2A",
        "agents": [research_agent.name, business_agent.name, engineering_agent.name],
        "questions": 3
    }


if __name__ == "__main__":
    demo_a2a()
