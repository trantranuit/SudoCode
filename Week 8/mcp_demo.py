"""
MCP (Model Context Protocol) Demo Implementation

MCP is a client-server architecture where:
- Server provides context, resources, and tools
- Client (AI model) requests context when needed
- Centralized context management
- One-way communication: client queries server, server responds
"""

import json
import time
import requests
import sqlite3
from typing import Dict, Any, List
from database import Database
import config


class MCPServer:
    """
    MCP Server: Provides context and resources to AI models.
    Acts as a centralized knowledge base - reads from SQLite database.
    """
    
    def __init__(self):
        self.db_path = config.DATABASE_PATH
        
        # Define available tools
        self.tools = {
            "get_company_info": self._get_company_info,
            "get_products": self._get_products,
            "get_financial_data": self._get_financial_data,
            "get_employees": self._get_employees,
            "search_database": self._search_database
        }
    
    def _get_db_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def _get_company_info(self) -> Dict[str, Any]:
        """Tool: Get company information from database."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name, founded, employees, focus FROM company_info LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "name": row[0],
                "founded": row[1],
                "employees": row[2],
                "focus": row[3]
            }
        return {}
    
    def _get_products(self) -> List[Dict[str, Any]]:
        """Tool: Get product list from database."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name, type, users, description FROM products")
        rows = cursor.fetchall()
        conn.close()
        
        products = []
        for row in rows:
            products.append({
                "name": row[0],
                "type": row[1],
                "users": row[2],
                "description": row[3]
            })
        return products
    
    def _get_financial_data(self) -> List[Dict[str, Any]]:
        """Tool: Get financial data from database."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT year, revenue, growth, funding FROM financial_data ORDER BY year DESC")
        rows = cursor.fetchall()
        conn.close()
        
        financial = []
        for row in rows:
            financial.append({
                "year": row[0],
                "revenue": row[1],
                "growth": row[2],
                "funding": row[3]
            })
        return financial
    
    def _get_employees(self) -> List[Dict[str, Any]]:
        """Tool: Get employees from database."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name, role, department, expertise FROM employees")
        rows = cursor.fetchall()
        conn.close()
        
        employees = []
        for row in rows:
            employees.append({
                "name": row[0],
                "role": row[1],
                "department": row[2],
                "expertise": row[3]
            })
        return employees
    
    def _search_database(self, query: str) -> Dict[str, Any]:
        """Tool: Search through database."""
        results = {}
        query_lower = query.lower()
        
        # Search in products
        if "product" in query_lower or "sản phẩm" in query_lower:
            results["products"] = self._get_products()
        
        # Search in financial data
        if "financial" in query_lower or "tài chính" in query_lower:
            results["financial_data"] = self._get_financial_data()
        
        # Search in employees
        if "employee" in query_lower or "nhân viên" in query_lower:
            results["employees"] = self._get_employees()
        
        return results
    
    def get_context(self, request: str) -> Dict[str, Any]:
        """
        Provide context based on client request.
        This is the core MCP function - server queries database and provides context to client.
        """
        context = {
            "available_tools": list(self.tools.keys()),
            "server_info": "MCP Server v2.0 - Database-backed",
            "data_source": "SQLite Database"
        }
        
        # Determine which resources are relevant to the request
        request_lower = request.lower()
        
        if "company" in request_lower or "doanh nghiệp" in request_lower or "công ty" in request_lower:
            context["company_info"] = self._get_company_info()
        
        if "product" in request_lower or "sản phẩm" in request_lower:
            context["products"] = self._get_products()
        
        if "financial" in request_lower or "tài chính" in request_lower or "doanh thu" in request_lower:
            context["financial_data"] = self._get_financial_data()
        
        if "employee" in request_lower or "nhân viên" in request_lower or "team" in request_lower or "phòng ban" in request_lower:
            context["employees"] = self._get_employees()
        
        return context


class MCPClient:
    """
    MCP Client: Requests context from server and uses it with AI model.
    Represents the AI model/application that needs context.
    """
    
    def __init__(self, server: MCPServer, database: Database):
        self.server = server
        self.database = database
        self.api_key = config.OPENAI_API_KEY
        self.api_url = config.OPENAI_API_URL
        self.model = config.OPENAI_MODEL
    
    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI API with messages."""
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
    
    def process_request(self, user_request: str) -> Dict[str, Any]:
        """
        Process user request using MCP pattern:
        1. Request context from server
        2. Combine context with user request
        3. Send to LLM
        4. Return response
        """
        start_time = time.time()
        
        # Step 1: Request context from MCP Server
        print(f"[Client] Requesting context...")
        print(f"[Server] Querying database...")
        server_context = self.server.get_context(user_request)
        
        # Step 2: Combine context with user request
        system_message = f"""You are an AI assistant with access to company data.
        
Available context from server:
{json.dumps(server_context, indent=2, ensure_ascii=False)}

Use this context to answer the user's question accurately."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_request}
        ]
        
        # Step 3: Call LLM with enriched context
        print(f"[Client] Calling OpenAI API...")
        model_response = self.call_llm(messages)
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Log to database
        self.database.log_mcp_conversation(
            client_request=user_request,
            server_context=server_context,
            model_response=model_response,
            metadata={"latency": latency}
        )
        
        self.database.log_metric(
            protocol_type="MCP",
            metric_name="request_latency",
            metric_value=latency
        )
        
        return {
            "protocol": "MCP",
            "request": user_request,
            "context_provided": server_context,
            "response": model_response,
            "latency": latency
        }


def demo_mcp():
    """Run MCP protocol demo."""
    print("\n" + "="*60)
    print("MCP (Model Context Protocol)")
    print("Client-Server: Server queries DB → provides context")
    print("="*60)
    
    db = Database()
    server = MCPServer()
    client = MCPClient(server, db)
    
    # Demo questions - SAME as A2A for comparison
    demo_questions = [
        "Cho tôi biết thông tin về công ty và các sản phẩm chính?",
        "Tình hình tài chính của công ty năm 2024 như thế nào?",
        "Công ty có bao nhiêu nhân viên và họ làm việc ở các phòng ban nào?"
    ]
    
    results = []
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")
        
        result = client.process_request(question)
        results.append(result)
        
        print(f"\nContext keys: {list(result['context_provided'].keys())}")
        print(f"A: {result['response']}")
        print(f"Time: {result['latency']:.2f}s")
        
        time.sleep(1)
    
    return results


if __name__ == "__main__":
    demo_mcp()
