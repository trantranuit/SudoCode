"""
Main Demo Script - MCP vs A2A Protocols

This script demonstrates the key differences between:
- MCP (Model Context Protocol): Client-Server architecture  
- A2A (Agent-to-Agent): Peer-to-Peer architecture

Flow:
1. Seed database with sample data
2. MCP Server queries database to provide context
3. A2A Agents query database directly for information
"""

import sys
import time
from database import Database
from mcp_demo import demo_mcp
from a2a_demo import demo_a2a
from seed_data import seed_database, view_data
import config


def print_header():
    """Print demo header."""
    print("\n" + "="*60)
    print("MCP vs A2A Protocol Demo")
    print("="*60)


def check_api_key():
    """Check if OpenAI API key is configured."""
    if config.OPENAI_API_KEY == "YOUR_API_KEY_HERE" or not config.OPENAI_API_KEY:
        print("\n❌ ERROR: Chưa cấu hình OpenAI API key!")
        print("Set: export OPENAI_API_KEY='sk-...'")
        print("Hoặc sửa trong config.py")
        sys.exit(1)
    
    print(f"✅ API Key configured - Model: {config.OPENAI_MODEL}")
    return True


def print_comparison_table():
    """Print comparison table between MCP and A2A."""
    print("\n" + "="*60)
    print("So sánh MCP vs A2A:")
    print("MCP: Client-Server, Server queries DB → Client")
    print("A2A: Peer-to-Peer, Agents query DB directly")
    print("\n⚠️  Cả 2 protocol sẽ trả lời CÙNG các câu hỏi để so sánh")
    print("="*60)


def show_database_stats(db: Database):
    """Show database statistics."""
    print("\n" + "="*60)
    print("DATABASE STATS & COMPARISON")
    print("="*60)
    
    mcp_history = db.get_mcp_history(limit=100)
    a2a_history = db.get_a2a_history(limit=100)
    
    print(f"\nConversations:")
    print(f"  MCP: {len(mcp_history)} requests")
    print(f"  A2A: {len(a2a_history)} messages")
    
    mcp_metrics = db.get_protocol_metrics("MCP")
    a2a_metrics = db.get_protocol_metrics("A2A")
    
    if mcp_metrics and a2a_metrics:
        avg_mcp = sum(m['metric_value'] for m in mcp_metrics) / len(mcp_metrics)
        avg_a2a = sum(m['metric_value'] for m in a2a_metrics) / len(a2a_metrics)
        
        print(f"\nAverage Latency:")
        print(f"  MCP: {avg_mcp:.2f}s")
        print(f"  A2A: {avg_a2a:.2f}s")
        
        faster = "MCP" if avg_mcp < avg_a2a else "A2A"
        diff = abs(avg_mcp - avg_a2a)
        print(f"\n→ {faster} is faster by {diff:.2f}s")
    
    print("\nKey Differences:")
    print("  MCP: Server queries DB once → Client uses context")
    print("  A2A: Each Agent queries DB independently")
    print("="*60)


def run_interactive_demo():
    """Run interactive demo mode."""
    db = Database()
    
    while True:
        print("\n" + "="*60)
        print("MENU")
        print("="*60)
        print("1. Seed Database")
        print("2. View Database")
        print("3. Run MCP Demo")
        print("4. Run A2A Demo")
        print("5. Run Both")
        print("6. View Stats")
        print("7. Clear Data")
        print("8. Exit")
        
        choice = input("\nChọn (1-8): ").strip()
        
        if choice == "1":
            seed_database()
        elif choice == "2":
            view_data()
        elif choice == "3":
            demo_mcp()
        elif choice == "4":
            demo_a2a()
        elif choice == "5":
            print("\n" + "="*60)
            print("MCP Demo")
            print("="*60)
            demo_mcp()
            time.sleep(2)
            print("\n" + "="*60)
            print("A2A Demo")
            print("="*60)
            demo_a2a()
        elif choice == "6":
            show_database_stats(db)
        elif choice == "7":
            confirm = input("Xóa data? (y/n): ").strip().lower()
            if confirm == 'y':
                db.clear_all_data()
                print("✓ Cleared!")
        elif choice == "8":
            print("Goodbye!")
            break
        else:
            print("Invalid option.")


def main():
    """Main function."""
    print_header()
    check_api_key()
    print_comparison_table()
    
    print("\n" + "="*60)
    print("MODE")
    print("="*60)
    print("1. Auto (Seed DB → MCP → A2A)")
    print("2. Interactive")
    
    mode = input("\nChọn (1/2): ").strip()
    
    if mode == "2":
        run_interactive_demo()
    else:
        print("\n▶ Auto Mode Started\n")
        db = Database()
        
        print("="*60)
        print("Seeding Database...")
        print("="*60)
        seed_database()
        time.sleep(2)
        
        print("\n="*60)
        print("MCP Demo")
        print("="*60)
        demo_mcp()
        time.sleep(2)
        
        print("\n="*60)
        print("A2A Demo")
        print("="*60)
        demo_a2a()
        time.sleep(2)
        
        show_database_stats(db)
        print(f"\n✅ Done! Database: {config.DATABASE_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
