import os
import logging
from pathlib import Path
import train_decentralized

def setup_logging():
    """Configure logging for the application."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'trading_system.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main entry point for the decentralized trading system."""
    logger = setup_logging()
    logger.info("Starting Decentralized Trading System")
    
    try:
        # This will automatically detect the role and run the appropriate logic
        train_decentralized.main()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.exception("Fatal error in main loop")
        raise

if __name__ == "__main__":
    main()
