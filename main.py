from smart_reply_system import SmartReplySystem
from utils import ensure_csv_headers, display_analysis_results
from config import initialize_nlp_resources, CUSTOM_STOPWORDS, CSV_FILES, CSV_HEADERS
import os
from datetime import datetime
from smart_reply_system import SmartReplySystem

def main():
    """
    Main function to run the conversational AI chat application.
    """
    print("=" * 60)
    print("      ADVANCED AI CONVERSATION SYSTEM      ")
    print("="*60)
    user_id = input("Enter your User ID: ").strip() or "guest"
    user_name = input("Enter your Name: ").strip() or "User"
    history_dir = "conversation_history"
    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_filename = os.path.join(history_dir, f"history_{user_id}_{timestamp}.json")
    smart_system = SmartReplySystem(
        user_id=user_id,
        user_name=user_name,
        history_filename=history_filename,
        prompt_style='enhanced'  # Or 'original'
    )

    print(f"\nWelcome, {user_name}! Start chatting (type 'exit' to quit).")
    if not smart_system.conversation_history:
        print("System: No prior conversation history found. Let's start fresh!\n")
    else:
        print("System: Loaded previous conversation history.\n")

    try:
        while True:
            user_input = input(f"{user_name}: ").strip()

            if user_input.lower() == 'exit':
                print("\nExiting conversation...")
                break

            if not user_input:
                continue
            analysis_result = smart_system.process_input(user_input)
            if analysis_result['reply_suggestions']:
                chosen_reply = smart_system.collect_and_log_feedback(
                    suggestions=analysis_result['reply_suggestions']
                )
                print(f"\nAI Assistant: {chosen_reply}")
            else:
                print("\nAI Assistant: I'm not sure how to respond to that. Can you rephrase?")

    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted by user. Exiting...")

    finally:
        smart_system.collect_exit_feedback()
        smart_system.save_conversation_history()
        smart_system.cleanup()  # Clean up database connections
        print(f"\nConversation history saved to {history_filename}")
        print("Goodbye👋! have a nice day😊")


if __name__ == "__main__":
    main()