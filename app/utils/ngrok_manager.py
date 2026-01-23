"""
Ngrok Manager for Public URL Tunneling
Handles ngrok tunnel creation and management for exposing local server
"""

from typing import Optional
import sys

class NgrokManager:
    """Manages ngrok tunnel for public access"""
    
    def __init__(self):
        """Initialize NgrokManager"""
        self.tunnel = None
        self.public_url: Optional[str] = None
        self.ngrok = None
    
    def start_tunnel(self, port: int, auth_token: Optional[str] = None) -> Optional[str]:
        """
        Start ngrok tunnel on specified port
        
        Args:
            port: Local port to tunnel
            auth_token: Optional ngrok auth token for custom domains/features
            
        Returns:
            Public URL if successful, None otherwise
        """
        try:
            # Import ngrok
            from pyngrok import ngrok, conf
            import nest_asyncio
            
            # Allow nested event loops (needed for Jupyter/Colab)
            nest_asyncio.apply()
            
            self.ngrok = ngrok
            
            # Set auth token if provided
            if auth_token:
                conf.get_default().auth_token = auth_token
                print("🔑 Ngrok auth token configured")
            
            # Start tunnel
            print(f"\n{'='*60}")
            print("🚀 Starting ngrok tunnel...")
            print(f"{'='*60}")
            
            # Create HTTP tunnel
            self.tunnel = ngrok.connect(
                port,
                bind_tls=True  # Force HTTPS
            )
            
            self.public_url = self.tunnel.public_url
            
            # Display tunnel information
            self._display_tunnel_info()
            
            return self.public_url
            
        except ImportError:
            self._handle_import_error()
            return None
            
        except Exception as e:
            self._handle_tunnel_error(e)
            return None
    
    def _display_tunnel_info(self) -> None:
        """Display tunnel information to user"""
        print(f"✅ Ngrok tunnel created successfully!")
        print(f"{'='*60}")
        print(f"📱 Public URL:      {self.public_url}")
        print(f"🌐 Ngrok Dashboard: http://127.0.0.1:4040")
        print(f"{'='*60}")
        print(f"\n💡 Share this URL with anyone to access your app:")
        print(f"   {self.public_url}")
        print(f"{'='*60}\n")
    
    def _handle_import_error(self) -> None:
        """Handle missing pyngrok/nest_asyncio packages"""
        print(f"\n{'='*60}")
        print("⚠️  WARNING: Ngrok dependencies not installed!")
        print(f"{'='*60}")
        print("\nTo enable ngrok tunneling, install:")
        print("  pip install pyngrok nest-asyncio")
        print("\nRunning without ngrok tunnel...")
        print(f"{'='*60}\n")
    
    def _handle_tunnel_error(self, error: Exception) -> None:
        """Handle ngrok tunnel creation errors"""
        print(f"\n{'='*60}")
        print("⚠️  ERROR: Failed to start ngrok tunnel!")
        print(f"{'='*60}")
        print(f"Error: {str(error)}\n")
        
        # Common error suggestions
        if "authtoken" in str(error).lower():
            print("💡 This might be an auth token issue.")
            print("   Get a free token at: https://dashboard.ngrok.com/signup")
            print("   Set it in config: NGROK_AUTH_TOKEN='your-token'\n")
        elif "429" in str(error) or "rate limit" in str(error).lower():
            print("💡 Rate limit reached. Try again in a few minutes")
            print("   or sign up for a free account at ngrok.com\n")
        
        print("Running without ngrok tunnel...")
        print(f"{'='*60}\n")
    
    def stop_tunnel(self) -> None:
        """Stop the ngrok tunnel gracefully"""
        if self.tunnel and self.ngrok:
            try:
                self.ngrok.disconnect(self.public_url)
                print("\n✅ Ngrok tunnel closed successfully")
            except Exception as e:
                print(f"\n⚠️ Error closing ngrok tunnel: {str(e)}")
        
        self.tunnel = None
        self.public_url = None
    
    def get_public_url(self) -> Optional[str]:
        """
        Get the current public URL
        
        Returns:
            Public URL string or None if not connected
        """
        return self.public_url
    
    def is_connected(self) -> bool:
        """
        Check if ngrok tunnel is active
        
        Returns:
            True if tunnel is active, False otherwise
        """
        return self.tunnel is not None and self.public_url is not None

# ==============================
# Global Ngrok Manager Instance
# ==============================

ngrok_manager = NgrokManager()