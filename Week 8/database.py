"""
Database setup for storing conversation history and protocol metrics.
Uses SQLite for lightweight data persistence.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any
import config


class Database:
    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database tables."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Table for MCP conversations (client-server interactions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mcp_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                client_request TEXT NOT NULL,
                server_context TEXT,
                model_response TEXT,
                protocol_type TEXT DEFAULT 'MCP',
                metadata TEXT
            )
        """)
        
        # Table for A2A conversations (agent-to-agent interactions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS a2a_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sender_agent TEXT NOT NULL,
                receiver_agent TEXT NOT NULL,
                message TEXT NOT NULL,
                response TEXT,
                protocol_type TEXT DEFAULT 'A2A',
                metadata TEXT
            )
        """)
        
        # Table for protocol metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS protocol_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                protocol_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_mcp_conversation(self, client_request: str, server_context: Dict[str, Any],
                            model_response: str, metadata: Dict[str, Any] = None):
        """Log MCP protocol conversation."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO mcp_conversations 
            (timestamp, client_request, server_context, model_response, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            client_request,
            json.dumps(server_context),
            model_response,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def log_a2a_conversation(self, sender: str, receiver: str,
                            message: str, response: str,
                            metadata: Dict[str, Any] = None):
        """Log A2A protocol conversation."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO a2a_conversations 
            (timestamp, sender_agent, receiver_agent, message, response, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            sender,
            receiver,
            message,
            response,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def log_metric(self, protocol_type: str, metric_name: str,
                   metric_value: float, metadata: Dict[str, Any] = None):
        """Log protocol performance metric."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO protocol_metrics 
            (timestamp, protocol_type, metric_name, metric_value, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            protocol_type,
            metric_name,
            metric_value,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_mcp_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get MCP conversation history."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM mcp_conversations 
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_a2a_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get A2A conversation history."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM a2a_conversations 
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_protocol_metrics(self, protocol_type: str = None) -> List[Dict[str, Any]]:
        """Get protocol performance metrics."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if protocol_type:
            cursor.execute("""
                SELECT * FROM protocol_metrics 
                WHERE protocol_type = ?
                ORDER BY timestamp DESC
            """, (protocol_type,))
        else:
            cursor.execute("""
                SELECT * FROM protocol_metrics 
                ORDER BY timestamp DESC
            """)
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def clear_all_data(self):
        """Clear all data from database (for testing)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM mcp_conversations")
        cursor.execute("DELETE FROM a2a_conversations")
        cursor.execute("DELETE FROM protocol_metrics")
        
        conn.commit()
        conn.close()


if __name__ == "__main__":
    # Test database setup
    db = Database()
    print("✓ Database initialized successfully!")
    print(f"✓ Database path: {config.DATABASE_PATH}")
