"""
Seed data into SQLite database for MCP and A2A demos.
This script populates the database with sample company data.
"""

import sqlite3
from datetime import datetime
import config


def seed_database():
    """Populate database with sample data."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create company_info table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS company_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            founded INTEGER,
            employees INTEGER,
            focus TEXT,
            created_at TEXT
        )
    """)
    
    # Create products table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT,
            users INTEGER,
            description TEXT,
            created_at TEXT
        )
    """)
    
    # Create financial_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS financial_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER,
            revenue TEXT,
            growth TEXT,
            funding TEXT,
            created_at TEXT
        )
    """)
    
    # Create employees table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT,
            department TEXT,
            expertise TEXT,
            created_at TEXT
        )
    """)
    
    # Clear existing data
    cursor.execute("DELETE FROM company_info")
    cursor.execute("DELETE FROM products")
    cursor.execute("DELETE FROM financial_data")
    cursor.execute("DELETE FROM employees")
    
    # Insert company info
    cursor.execute("""
        INSERT INTO company_info (name, founded, employees, focus, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, ("TechCorp Vietnam", 2020, 150, "AI và Machine Learning", datetime.now().isoformat()))
    
    # Insert products
    products_data = [
        ("SmartChat", "Chatbot", 50000, "AI-powered chatbot platform với NLP capabilities", datetime.now().isoformat()),
        ("DataViz Pro", "Analytics", 25000, "Data visualization và business intelligence tool", datetime.now().isoformat()),
        ("AutoML Suite", "ML Platform", 10000, "Automated machine learning platform cho enterprise", datetime.now().isoformat()),
    ]
    cursor.executemany("""
        INSERT INTO products (name, type, users, description, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, products_data)
    
    # Insert financial data
    financial_records = [
        (2024, "$5.2M", "145%", "Series A - $10M", datetime.now().isoformat()),
        (2023, "$2.1M", "230%", "Seed - $2M", datetime.now().isoformat()),
    ]
    cursor.executemany("""
        INSERT INTO financial_data (year, revenue, growth, funding, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, financial_records)
    
    # Insert employees
    employees_data = [
        ("Dr. Nguyen Van A", "Research Scientist", "AI Research", "Machine Learning, Deep Learning, NLP", datetime.now().isoformat()),
        ("Ms. Tran Thi B", "Business Strategist", "Business Development", "Market Analysis, Strategic Planning, Sales", datetime.now().isoformat()),
        ("Mr. Le Van C", "Lead Engineer", "Engineering", "System Architecture, Cloud Infrastructure, DevOps", datetime.now().isoformat()),
        ("Dr. Pham Thi D", "Data Scientist", "AI Research", "Data Analysis, Statistical Modeling, AI Ethics", datetime.now().isoformat()),
    ]
    cursor.executemany("""
        INSERT INTO employees (name, role, department, expertise, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, employees_data)
    
    conn.commit()
    
    print("="*60)
    print("✅ Database seeded!")
    print("="*60)
    
    cursor.execute("SELECT COUNT(*) FROM company_info")
    print(f"Company: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM products")
    print(f"Products: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM financial_data")
    print(f"Financial: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM employees")
    print(f"Employees: {cursor.fetchone()[0]}")
    print("="*60)
    
    conn.close()


def view_data():
    """View seeded data."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("COMPANY INFO")
    print("="*60)
    cursor.execute("SELECT * FROM company_info")
    for row in cursor.fetchall():
        print(f"{row[1]}, Founded: {row[2]}, Employees: {row[3]}, Focus: {row[4]}")
    
    print("\n" + "="*60)
    print("PRODUCTS")
    print("="*60)
    cursor.execute("SELECT * FROM products")
    for row in cursor.fetchall():
        print(f"{row[1]} ({row[2]}) - {row[3]} users")
    
    print("\n" + "="*60)
    print("FINANCIAL DATA")
    print("="*60)
    cursor.execute("SELECT * FROM financial_data")
    for row in cursor.fetchall():
        print(f"{row[1]}: Revenue {row[2]}, Growth {row[3]}")
    
    print("\n" + "="*60)
    print("EMPLOYEES")
    print("="*60)
    cursor.execute("SELECT * FROM employees")
    for row in cursor.fetchall():
        print(f"{row[1]} - {row[2]}")
    print("="*60)
    
    conn.close()


if __name__ == "__main__":
    seed_database()
    view_data()