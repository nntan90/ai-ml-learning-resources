# Notebook: 15-browser-user

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/15-browser-use/15-browser-user.ipynb

---

# Finding the Cheapest Airbnb with AI-Powered Web Automation

This notebook demonstrates how to build an intelligent web automation agent that searches Airbnb, extracts prices, and finds the cheapest listing in Stockholm. You'll learn how to integrate **Playwright** with **Browser-Use** for powerful AI-driven automation.

## What You'll Learn:
1. **Playwright + Browser-Use Integration**: Combining browser management with AI automation
2. **Vision-Based Price Extraction**: Let AI "see" and read prices from web pages
3. **Structured Data Extraction**: Extract listing data with type-safe Pydantic models
4. **Price Comparison Logic**: Find the cheapest option from multiple listings
5. **Real-world Application**: Practical price comparison automation

## Prerequisites:
- Azure OpenAI deployment configured
- Playwright installed (`pip install playwright`)
- Understanding of async Python
- Basic web automation concepts

## Understanding the Playwright + Browser-Use Architecture

This notebook uses the **official Playwright integration** pattern from Browser-Use documentation.

### Architecture Flow:
```
┌──────────────────┐
│   Playwright     │ ◄─── Manages browser lifecycle
│  Browser Manager │      Handles CDP connection
└────────┬─────────┘      Provides browser instance
         │
         │ playwright_browser parameter
         ▼
┌──────────────────┐
│  Browser-Use     │ ◄─── AI-powered automation
│  Browser Object  │      Wraps Playwright browser
└────────┬─────────┘      Provides Agent interface
         │
         │ uses
         ▼
┌──────────────────┐
│   Agent          │ ◄─── Vision + Decision Making
│ (with LLM)       │      Structured output extraction
└──────────────────┘      Natural language tasks
         │
         │ powered by
         ▼
┌──────────────────┐
│  Azure OpenAI    │ ◄─── GPT-4 Vision
│  (LLM + Vision)  │      Analyzes screenshots
└──────────────────┘      Extracts structured data
```

### Why This Approach?

**Playwright provides:**
- ✅ Robust browser lifecycle management
- ✅ Full Chrome DevTools Protocol control
- ✅ Stable page and context handling
- ✅ Built-in waiting and synchronization

**Browser-Use adds:**
- ✅ AI-powered element finding (no CSS selectors needed!)
- ✅ Vision-based page understanding
- ✅ Structured output extraction with Pydantic
- ✅ Natural language task execution

**Together they enable:**
- 🎯 "Search for Stockholm Airbnb" → Agent navigates
- 👁️ Vision reads all prices on the page
- 📊 Structured extraction → clean Python objects
- 💰 Price comparison logic → find cheapest

### Our Task Flow:
1. **Playwright** launches Chrome browser
2. **Browser-Use Agent** navigates to Airbnb.com
3. **Agent searches** for "Stockholm, Sweden"
4. **Vision model** reads and extracts all listing prices
5. **Structured output** returns typed data (Pydantic models)
6. **Python code** compares prices and finds cheapest
7. **Display results** with rich formatting

```python
pip install browser_use langchain-openai playwright 
```

```python
!playwright install chromium
```

```python
import asyncio
import os
import re
from typing import Optional, List
from IPython.display import display, HTML, Markdown
from dotenv import load_dotenv

# Playwright imports
from playwright.async_api import async_playwright

# Browser-Use imports - USE BROWSER-USE'S AZURE OPENAI!
# Changed from langchain_openai
from browser_use import Agent, Browser, ChatAzureOpenAI
from pydantic import BaseModel, Field

print("✅ All packages imported successfully")
```

```python
# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
azure_openai_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Verify configuration
print("✅ Azure OpenAI Configuration:")
print(f"   Endpoint: {azure_openai_endpoint}")
print(f"   Deployment: {azure_openai_deployment}")
print(f"   API Version: {api_version}")
```

## Initialize Azure OpenAI LLM

The LLM powers the Agent's decision-making and vision capabilities. We use:
- **Temperature: 0.3** for consistent, predictable automation
- **Vision capabilities** to "see" and understand page content
- **Structured output** to extract data into Pydantic models

```python
# Initialize Azure OpenAI with Browser-Use's ChatAzureOpenAI
llm = ChatAzureOpenAI(
    # Your deployment name (e.g., 'gpt-4o', 'gpt-4.1-mini')
    model=azure_openai_deployment,
    # Browser-Use reads these from environment variables automatically:
    # AZURE_OPENAI_ENDPOINT
    # AZURE_OPENAI_API_KEY
    # AZURE_OPENAI_API_VERSION (optional, defaults to latest)
)

print("✅ LLM initialized successfully")
print(f"   Model: {azure_openai_deployment}")
print(f"   Endpoint: {azure_openai_endpoint}")
print(f"   Integration: Browser-Use ChatAzureOpenAI")
```

## Define Structured Output Models

We use Pydantic models to extract structured data from Airbnb search results. The Agent will use GPT-4 Vision to read the page and extract data into these models automatically.

This ensures type safety and validation of all extracted data.

```python
# UPDATE THIS CELL - Add URL field to AirbnbListing
class AirbnbListing(BaseModel):
    """Single Airbnb listing with price information"""
    title: str = Field(description="Name/title of the listing")
    price_per_night: float = Field(
        description="Price per night as a number (extract just the numeric value, ignore currency symbols)")
    currency: str = Field(
        default="SEK", description="Currency code (SEK for Swedish Krona)")
    rating: Optional[float] = Field(
        default=None, description="Rating score if visible")
    url: Optional[str] = Field(
        default=None, description="Full URL link to the listing page")  # ✅ NEW!


class SearchResult(BaseModel):
    """Complete search results from Airbnb"""
    location: str = Field(description="Search location (Stockholm, Sweden)")
    total_listings_found: int = Field(
        description="Number of listings found on the page")
    listings: List[AirbnbListing] = Field(
        description="List of all listings with prices extracted from the page")
    cheapest_listing: AirbnbListing = Field(
        description="The listing with the lowest price per night")
    average_price: float = Field(
        description="Average price per night across all listings")
    price_range: str = Field(description="Price range as 'min - max SEK'")


print("✅ Structured output models defined")
print("   AirbnbListing: Individual listing data with clickable URLs")
print("   SearchResult: Complete search results with price analysis")
```

## Helper Functions for Display

These functions provide rich, educational output in the notebook with formatted HTML.



```python
class ListingInfo(BaseModel):
    """Information about the Airbnb listing"""
    title: str = Field(description="The name/title of the listing")
    location: str = Field(description="City and country of the listing")
    price_per_night: Optional[str] = Field(
        description="Price per night if visible")
    rating: Optional[str] = Field(description="Rating score if visible")


class BookingDates(BaseModel):
    """Selected booking dates"""
    check_in: str = Field(
        description="Check-in date in format: Month DD, YYYY")
    check_out: str = Field(
        description="Check-out date in format: Month DD, YYYY")
    nights: int = Field(description="Number of nights")


class BookingResult(BaseModel):
    """Complete booking result information"""
    success: bool = Field(description="Whether the booking flow was completed")
    listing_info: Optional[ListingInfo] = Field(
        description="Details about the listing")
    booking_dates: Optional[BookingDates] = Field(description="Selected dates")
    total_price: Optional[str] = Field(description="Total price if shown")
    message: str = Field(description="Status message or error description")


print("✅ Structured output models defined")
print("   ListingInfo: Extract listing details")
print("   BookingDates: Capture selected dates")
print("   BookingResult: Final booking status")
```

## Helper Functions for Display

These functions provide rich, educational output in the notebook.

```python
def display_step(step_number: int, title: str, description: str, color: str = "#2E8B57"):
    """Display a workflow step with formatting"""
    html = f"""
    <div style='
        margin: 20px 0; 
        padding: 20px; 
        border-left: 5px solid {color}; 
        background: linear-gradient(to right, rgba(46, 139, 87, 0.05), transparent); 
        border-radius: 8px;
    '>
        <h3 style='color: {color}; margin: 0 0 10px 0;'>
            Step {step_number}: {title}
        </h3>
        <p style='margin: 0; line-height: 1.6; color: #333;'>{description}</p>
    </div>
    """
    display(HTML(html))


def display_action(action_type: str, details: str, emoji: str = "⚙️"):
    """Display an action being performed"""
    html = f"""
    <div style='
        margin: 10px 20px;
        padding: 12px 16px;
        background: rgba(0, 123, 255, 0.05);
        border: 1px solid #007BFF;
        border-radius: 6px;
        font-family: monospace;
        font-size: 14px;
    '>
        <strong style='color: #007BFF;'>{emoji} {action_type}:</strong>
        <span style='color: #555; margin-left: 10px;'>{details}</span>
    </div>
    """
    display(HTML(html))


def display_result(success: bool, message: str):
    """Display a result with success/failure indication"""
    color = "#28a745" if success else "#dc3545"
    emoji = "✅" if success else "❌"
    html = f"""
    <div style='
        margin: 20px 0;
        padding: 15px 20px;
        border-left: 5px solid {color};
        background: rgba({"40, 167, 69" if success else "220, 53, 69"}, 0.1);
        border-radius: 8px;
    '>
        <strong style='color: {color}; font-size: 16px;'>{emoji} {message}</strong>
    </div>
    """
    display(HTML(html))


def display_screenshot(screenshot_base64: str, caption: str = ""):
    """Display a screenshot in the notebook"""
    html = f"""
    <div style='margin: 20px 0; text-align: center;'>
        <img src='data:image/png;base64,{screenshot_base64}' 
             style='max-width: 100%; border: 2px solid #ddd; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'/>
        {f"<p style='margin-top: 10px; color: #666; font-style: italic;'>{caption}</p>" if caption else ""}
    </div>
    """
    display(HTML(html))


print("✅ Helper functions loaded")
```

## The Airbnb Booking Agent Class

This class orchestrates the entire booking workflow, combining Agent and Actor approaches strategically.

```python
class AirbnbSearchAgent:
    """
    Intelligent Airbnb search agent using Playwright + Browser-Use integration.
    
    This agent:
    1. Navigates to Airbnb.com
    2. Searches for listings in Stockholm
    3. Extracts all visible prices using AI vision
    4. Compares prices and finds the cheapest listing
    """
    
    def __init__(self, llm, playwright_browser):
        self.llm = llm
        # Browser-Use wraps the Playwright browser
        # Reference: https://docs.browser-use.com/examples/templates/playwright-integration
        self.browser = Browser(playwright_browser=playwright_browser)
        
    async def take_screenshot(self, caption: str = ""):
        """Take and display a screenshot of current page"""
        try:
            page = await self.browser.get_current_page()
            screenshot_base64 = await page.screenshot(format='png')
            display_screenshot(screenshot_base64, caption)
        except Exception as e:
            print(f"⚠️  Could not capture screenshot: {str(e)}")
    
    async def search_stockholm(self) -> SearchResult:
        """
        Main workflow: Search Airbnb for Stockholm and find cheapest listing.
        
        Returns:
            SearchResult: Structured data with all listings and price analysis
        """
        
        # Step 1: Navigate and search
        display_step(
            1, 
            "Navigate & Search (AI Agent)", 
            "Using AI agent with vision to navigate Airbnb and search for Stockholm listings. "
            "The agent will handle pop-ups, cookie banners, and search automatically."
        )
        
        try:
            # Agent navigates to Airbnb and searches
            search_agent = Agent(
                task=(
                    "Navigate to https://www.airbnb.com. "
                    "Close any pop-ups, cookie banners, or login prompts if they appear. "
                    "Search for 'Stockholm, Sweden' in the search box. "
                    "Wait for the search results page to fully load with listing cards visible."
                ),
                llm=self.llm,
                browser=self.browser,
                use_vision=True  # Critical: enables screenshot analysis
            )
            
            display_action("Agent", "Navigating to Airbnb and searching for Stockholm...")
            await search_agent.run()
            
            # Wait for results to load
            await asyncio.sleep(3)
            await self.take_screenshot("Search results page loaded")
            
            display_result(True, "Successfully loaded Stockholm search results")
            
        except Exception as e:
            display_result(False, f"Search failed: {str(e)}")
            raise
        
        # Step 2: Extract prices with vision
        display_step(
            2,
            "Extract Prices (Vision + LLM)",
            "Using GPT-4 Vision to read all listing prices from the page and extract "
            "structured data into Pydantic models. The AI 'sees' the page like a human."
        )
        
        try:
            page = await self.browser.get_current_page()
            
            display_action("Vision", "AI is analyzing the page and reading prices...")
            
            # Use page.extract_content with structured output
            # Reference: https://docs.browser-use.com/customize/actor/all-parameters
            # This uses LLM to parse the visible page content
            extraction_prompt = """
            Extract ALL Airbnb listings visible on this page.
            
            For each listing, extract:
            - Title/name of the property
            - Price per night (numeric value only, without currency symbols)
            - Currency (SEK for Swedish Krona)
            - Rating if visible
            
            After extracting all listings:
            - Identify which listing has the LOWEST price
            - Calculate the average price across all listings
            - Determine the price range (min to max)
            
            Focus on the listing cards on the search results page.
            Only include listings where you can clearly see the price.
            """
            
            # Extract structured data using LLM vision
            search_results = await page.extract_content(
                prompt=extraction_prompt,
                structured_output=SearchResult,
                llm=self.llm
            )
            
            display_action(
                "Extracted", 
                f"Found {search_results.total_listings_found} listings with prices"
            )
            
            # Display some sample prices
            if len(search_results.listings) > 0:
                sample_prices = [f"{l.price_per_night:.0f} SEK" for l in search_results.listings[:5]]
                display_action(
                    "Sample Prices", 
                    f"{', '.join(sample_prices)}{'...' if len(search_results.listings) > 5 else ''}"
                )
            
            display_result(True, "Price extraction completed successfully")
            
            return search_results
            
        except Exception as e:
            display_result(False, f"Price extraction failed: {str(e)}")
            raise

print("✅ AirbnbSearchAgent class defined")
print("   Integration: Playwright browser + Browser-Use Agent")
print("   Capabilities: Vision-based price extraction with structured output")
```

## Execute the Search

Now let's run the complete workflow and find the cheapest Airbnb in Stockholm!

This will:
1. Launch a real Chrome browser (visible)
2. Use AI to navigate and search
3. Extract all prices using vision
4. Display the cheapest option

```python
# UPDATED AirbnbSearchAgent class - Add keep_alive parameter
class AirbnbSearchAgent:
    """
    Intelligent Airbnb search agent using Browser-Use integration via CDP.
    
    This agent:
    1. Connects to Chrome via CDP (Chrome DevTools Protocol)
    2. Both Playwright and Browser-Use share the same browser instance
    3. Searches for listings in Stockholm
    4. Extracts prices using AI vision
    5. Finds the cheapest listing
    """

    def __init__(self, llm, cdp_url: str):
        """
        Initialize agent with LLM and CDP connection.
        
        Args:
            llm: Language model for AI decisions
            cdp_url: Chrome DevTools Protocol URL (e.g., 'http://localhost:9222')
        """
        self.llm = llm
        # Browser-Use connects to Chrome via CDP
        # IMPORTANT: keep_alive=True prevents browser from closing after Agent completes
        self.browser = Browser(
            cdp_url=cdp_url,
            keep_alive=True  # ✅ This keeps the browser open!
        )

    async def take_screenshot(self, caption: str = ""):
        """Take and display a screenshot of current page"""
        try:
            # Get pages and use the active one
            pages = await self.browser.get_pages()
            if not pages:
                print("⚠️  No pages available for screenshot")
                return

            page = pages[0]  # Use first page (active page)
            screenshot_bytes = await page.screenshot()
            import base64
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode()
            display_screenshot(screenshot_base64, caption)
        except Exception as e:
            print(f"⚠️  Could not capture screenshot: {str(e)}")

    async def search_stockholm(self) -> SearchResult:
        """
        Main workflow: Search Airbnb for Stockholm and find cheapest listing.
        
        Returns:
            SearchResult: Structured data with all listings and price analysis
        """

        # Step 1: Navigate and search
        display_step(
            1,
            "Navigate & Search (AI Agent)",
            "Using AI agent with vision to navigate Airbnb and search for Stockholm listings. "
            "The agent will handle pop-ups, cookie banners, and search automatically."
        )

        try:
            # Agent navigates to Airbnb and searches
            search_agent = Agent(
                task=(
                    "Navigate to https://www.airbnb.com. "
                    "Close any pop-ups, cookie banners, or login prompts if they appear. "
                    "Search for 'Stockholm, Sweden' in the search box. "
                    "Wait for the search results page to fully load with listing cards visible."
                ),
                llm=self.llm,
                browser=self.browser,
                use_vision=True  # Critical: enables screenshot analysis
            )

            display_action(
                "Agent", "Navigating to Airbnb and searching for Stockholm...")
            await search_agent.run()

            # Wait for results to load
            await asyncio.sleep(3)
            await self.take_screenshot("Search results page loaded")

            display_result(
                True, "Successfully loaded Stockholm search results")

        except Exception as e:
            display_result(False, f"Search failed: {str(e)}")
            raise

        # Step 2: Extract prices with vision
        display_step(
            2,
            "Extract Prices (Vision + LLM)",
            "Using GPT-4 Vision to read all listing prices from the page and extract "
            "structured data into Pydantic models. The AI 'sees' the page like a human."
        )

        try:
            # Get the pages created by the Agent
            pages = await self.browser.get_pages()

            if not pages:
                raise RuntimeError(
                    "No pages available after Agent run. Browser might have closed.")

            # Use the first (active) page
            page = pages[0]

            display_action(
                "Vision", "AI is analyzing the page and reading prices...")

            # Extract structured data using LLM vision
            extraction_prompt = """
Extract ALL Airbnb Home listings visible on this page. DO NOT include "Experiences" or other non-home listings.

For each listing, extract:
- Title/name of the property
- Price per night (numeric value only, without currency symbols)
- Currency (SEK for Swedish Krona)
- Rating if visible
- URL: The full link to the listing detail page (should start with https://www.airbnb.com/rooms/)

After extracting all listings:
- Identify which listing has the LOWEST price
- Calculate the average price across all listings
- Determine the price range (min to max)

Focus on the listing cards on the search results page.
Only include listings where you can clearly see the price.
IMPORTANT: Extract the actual URL/link for each listing so users can click on it.
"""

            search_results = await page.extract_content(
                prompt=extraction_prompt,
                structured_output=SearchResult,
                llm=self.llm
            )

            display_action(
                "Extracted",
                f"Found {search_results.total_listings_found} listings with prices"
            )

            # Display some sample prices
            if len(search_results.listings) > 0:
                sample_prices = [
                    f"{l.price_per_night:.0f} SEK" for l in search_results.listings[:5]]
                display_action(
                    "Sample Prices",
                    f"{', '.join(sample_prices)}{'...' if len(search_results.listings) > 5 else ''}"
                )

            display_result(True, "Price extraction completed successfully")

            return search_results

        except Exception as e:
            display_result(False, f"Price extraction failed: {str(e)}")
            raise


print("✅ AirbnbSearchAgent class defined")
print("   Integration: CDP-based connection for Playwright + Browser-Use")
print("   Capabilities: Vision-based price extraction with structured output")
print("   Browser Mode: keep_alive=True (prevents auto-close)")
```

```python
import subprocess
import tempfile


async def start_chrome_with_cdp(port: int = 9222):
    """
    Start Chrome with CDP (Chrome DevTools Protocol) enabled.
    Returns the Chrome process.
    """
    # Create temporary directory for Chrome user data
    user_data_dir = tempfile.mkdtemp(prefix='chrome_cdp_')

    # Chrome paths for different platforms
    chrome_paths = [
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',  # macOS
        '/usr/bin/google-chrome',  # Linux
        '/usr/bin/chromium-browser',  # Linux Chromium
        'chrome',  # Windows/PATH
        'chromium',  # Generic
    ]

    chrome_exe = None
    for path in chrome_paths:
        if os.path.exists(path) or path in ['chrome', 'chromium']:
            try:
                test_proc = await asyncio.create_subprocess_exec(
                    path, '--version',
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                await test_proc.wait()
                chrome_exe = path
                break
            except Exception:
                continue

    if not chrome_exe:
        raise RuntimeError(
            '❌ Chrome not found. Please install Chrome or Chromium.')

    # Chrome command arguments
    cmd = [
        chrome_exe,
        f'--remote-debugging-port={port}',
        f'--user-data-dir={user_data_dir}',
        '--no-first-run',
        '--no-default-browser-check',
        'about:blank',
    ]

    # Start Chrome process
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for Chrome to start and CDP to be ready
    import aiohttp
    cdp_ready = False
    for _ in range(20):  # 20 second timeout
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'http://localhost:{port}/json/version',
                    timeout=aiohttp.ClientTimeout(total=1)
                ) as response:
                    if response.status == 200:
                        cdp_ready = True
                        break
        except Exception:
            pass
        await asyncio.sleep(1)

    if not cdp_ready:
        process.terminate()
        raise RuntimeError('❌ Chrome failed to start with CDP')

    print(f"✅ Chrome started with CDP on port {port}")
    return process


async def main():
    """
    Main execution function for Airbnb price comparison.
    
    Uses CDP to connect both Playwright and Browser-Use to the same Chrome instance.
    """

    display(HTML("""
    <div style='
        padding: 30px;
        background: linear-gradient(135deg, #FF5A5F 0%, #FF385C 100%);
        color: white;
        border-radius: 12px;
        margin: 30px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    '>
        <h2 style='margin: 0 0 15px 0;'>🏠 Find the Cheapest Airbnb in Stockholm</h2>
        <p style='margin: 0; font-size: 16px; line-height: 1.8;'>
            This demo uses <strong>CDP (Chrome DevTools Protocol)</strong> integration:<br>
            • <strong>Chrome</strong> runs with remote debugging enabled<br>
            • <strong>Playwright</strong> connects to Chrome via CDP<br>
            • <strong>Browser-Use</strong> connects to same Chrome via CDP<br>
            • <strong>GPT-4 Vision</strong> reads and extracts prices<br>
            • <strong>Structured Output</strong> returns type-safe data
        </p>
        <p style='margin: 15px 0 0 0; font-size: 14px; opacity: 0.9;'>
            📊 Watch the browser as the AI agent searches and analyzes prices!
        </p>
    </div>
    """))

    chrome_process = None
    playwright_browser = None

    try:
        # Step 1: Start Chrome with CDP
        display_action(
            "Chrome", "Starting Chrome with CDP (remote debugging)...")
        chrome_process = await start_chrome_with_cdp(port=9222)
        cdp_url = 'http://localhost:9222'

        # Step 2: Connect Playwright to CDP (optional - for custom Playwright actions)
        display_action(
            "Playwright", "Connecting Playwright to Chrome via CDP...")
        playwright = await async_playwright().start()
        playwright_browser = await playwright.chromium.connect_over_cdp(cdp_url)
        display_result(True, "Playwright connected successfully")

        # Step 3: Create Browser-Use agent with CDP connection
        display_action(
            "Browser-Use", "Creating Browser-Use agent with CDP connection...")
        agent = AirbnbSearchAgent(llm=llm, cdp_url=cdp_url)
        display_result(True, "Agent initialized with CDP integration")

        # Step 4: Search and extract prices
        result = await agent.search_stockholm()

        # Step 5: Display results
        display(
            HTML("<hr style='margin: 40px 0; border: none; border-top: 2px solid #ddd;'>"))

        display(HTML("""
        <div style='padding: 20px; background: #f8f9fa; border-radius: 8px; margin: 20px 0;'>
            <h3 style='margin: 0 0 15px 0; color: #333;'>📊 Search Results</h3>
        </div>
        """))

        # Display summary stats
        display(HTML(f"""
        <div style='margin: 20px; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h4 style='margin: 0 0 15px 0; color: #FF5A5F;'>📈 Price Analysis</h4>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr>
                    <td style='padding: 10px; border-bottom: 1px solid #eee; font-weight: bold;'>Location:</td>
                    <td style='padding: 10px; border-bottom: 1px solid #eee;'>{result.location}</td>
                </tr>
                <tr>
                    <td style='padding: 10px; border-bottom: 1px solid #eee; font-weight: bold;'>Total Listings Found:</td>
                    <td style='padding: 10px; border-bottom: 1px solid #eee;'>{result.total_listings_found}</td>
                </tr>
                <tr>
                    <td style='padding: 10px; border-bottom: 1px solid #eee; font-weight: bold;'>Average Price:</td>
                    <td style='padding: 10px; border-bottom: 1px solid #eee;'>{result.average_price:.2f} SEK/night</td>
                </tr>
                <tr>
                    <td style='padding: 10px; border-bottom: 1px solid #eee; font-weight: bold;'>Price Range:</td>
                    <td style='padding: 10px; border-bottom: 1px solid #eee;'>{result.price_range}</td>
                </tr>
            </table>
        </div>
        """))

        # Display the CHEAPEST listing with clickable link
        cheapest = result.cheapest_listing

        # Create View Listing button if URL exists
        view_button = ""
        if cheapest.url:
            view_button = f"""
            <a href="{cheapest.url}" target="_blank" style="
                display: inline-block;
                margin-top: 15px;
                padding: 12px 24px;
                background: #FF5A5F;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                font-weight: bold;
                transition: background 0.3s;
            " onmouseover="this.style.background='#E00007'" onmouseout="this.style.background='#FF5A5F'">
                🔗 View Listing on Airbnb
            </a>
            """

        display(HTML(f"""
        <div style='
            margin: 20px; 
            padding: 25px; 
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            border-radius: 12px; 
            box-shadow: 0 4px 12px rgba(255,165,0,0.3);
            border: 3px solid #FFD700;
        '>
            <h3 style='margin: 0 0 15px 0; color: #333; font-size: 24px;'>
                🏆 CHEAPEST AIRBNB IN STOCKHOLM
            </h3>
            <div style='background: white; padding: 20px; border-radius: 8px; margin-top: 15px;'>
                <h4 style='margin: 0 0 10px 0; color: #FF5A5F;'>{cheapest.title}</h4>
                <p style='margin: 10px 0; font-size: 28px; font-weight: bold; color: #28a745;'>
                    {cheapest.price_per_night:.2f} {cheapest.currency}/night
                </p>
                {f"<p style='margin: 10px 0; color: #666;'>⭐ Rating: {cheapest.rating}/5.0</p>" if cheapest.rating else ""}
                <p style='margin: 15px 0 0 0; color: #666; font-size: 14px;'>
                    💰 Saves you <strong>{(result.average_price - cheapest.price_per_night):.2f} SEK</strong> compared to average price!
                </p>
                {view_button}
            </div>
        </div>
        """))

        # Display all listings in a clickable table
        if len(result.listings) > 1:
            listings_html = """
            <div style='margin: 20px; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <h4 style='margin: 0 0 15px 0; color: #FF5A5F;'>📋 All Listings Found (Click to View)</h4>
                <table style='width: 100%; border-collapse: collapse;'>
                    <thead>
                        <tr style='background: #f8f9fa;'>
                            <th style='padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;'>Rank</th>
                            <th style='padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;'>Listing</th>
                            <th style='padding: 12px; text-align: right; border-bottom: 2px solid #dee2e6;'>Price/Night</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Rating</th>
                            <th style='padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;'>Link</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            sorted_listings = sorted(
                result.listings, key=lambda x: x.price_per_night)

            for idx, listing in enumerate(sorted_listings, 1):
                is_cheapest = listing.price_per_night == cheapest.price_per_night
                row_style = "background: #fff3cd;" if is_cheapest else ""
                badge = "🏆 " if is_cheapest else ""

                # Create clickable link if URL exists
                if listing.url:
                    link_html = f'<a href="{listing.url}" target="_blank" style="color: #FF5A5F; text-decoration: none; font-weight: bold; padding: 6px 12px; border: 1px solid #FF5A5F; border-radius: 4px; transition: all 0.3s;" onmouseover="this.style.background=\'#FF5A5F\'; this.style.color=\'white\'" onmouseout="this.style.background=\'transparent\'; this.style.color=\'#FF5A5F\'">🔗 View</a>'
                    title_html = f'<a href="{listing.url}" target="_blank" style="color: #333; text-decoration: none; font-weight: 500;" onmouseover="this.style.color=\'#FF5A5F\'" onmouseout="this.style.color=\'#333\'">{listing.title[:50]}...</a>'
                else:
                    link_html = '<span style="color: #999;">N/A</span>'
                    title_html = f'{listing.title[:50]}...'

                listings_html += f"""
                    <tr style='{row_style}'>
                        <td style='padding: 10px; border-bottom: 1px solid #eee;'>{badge}{idx}</td>
                        <td style='padding: 10px; border-bottom: 1px solid #eee;'>{title_html}</td>
                        <td style='padding: 10px; border-bottom: 1px solid #eee; text-align: right; font-weight: bold;'>
                            {listing.price_per_night:.2f} {listing.currency}
                        </td>
                        <td style='padding: 10px; border-bottom: 1px solid #eee; text-align: center;'>
                            {f"⭐ {listing.rating}" if listing.rating else "N/A"}
                        </td>
                        <td style='padding: 10px; border-bottom: 1px solid #eee; text-align: center;'>
                            {link_html}
                        </td>
                    </tr>
                """

            listings_html += """
                    </tbody>
                </table>
                <p style='margin-top: 15px; color: #666; font-size: 13px; font-style: italic;'>
                    💡 Tip: Click on any listing title or the "View" button to open it in a new tab
                </p>
            </div>
            """

            display(HTML(listings_html))

        display_result(True, "Price comparison completed successfully!")

    except Exception as e:
        display_result(False, f"Error during search: {str(e)}")
        import traceback
        print(traceback.format_exc())

    finally:
        # Cleanup
        display_action("Cleanup", "Closing browser and cleaning up...")

        if playwright_browser:
            await playwright_browser.close()

        if chrome_process:
            chrome_process.terminate()
            try:
                await asyncio.wait_for(chrome_process.wait(), 5)
            except TimeoutError:
                chrome_process.kill()

        display_result(True, "Cleanup complete")

    # Display educational summary
    display(HTML("""
    <div style='
        margin: 40px 0 20px 0;
        padding: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
    '>
        <h3 style='margin: 0 0 15px 0;'>🎓 What You Just Learned</h3>
        <ul style='margin: 0; padding-left: 20px; line-height: 2;'>
            <li><strong>CDP Integration:</strong> Chrome DevTools Protocol connects Playwright + Browser-Use</li>
            <li><strong>Vision-Based Extraction:</strong> GPT-4 Vision reads prices like a human</li>
            <li><strong>Structured Output:</strong> Type-safe data extraction with Pydantic models</li>
            <li><strong>Price Comparison:</strong> Automated analysis to find best deals</li>
            <li><strong>Clickable URLs:</strong> Direct links to Airbnb listings for easy viewing</li>
            <li><strong>Real-World Application:</strong> Practical web scraping for price monitoring</li>
        </ul>
    </div>
    """))

# Run the demo
await main()
```

## Key Takeaways and Best Practices

### When to Use Agent vs Actor

| Scenario | Use Agent | Use Actor |
|----------|-----------|-----------|
| **Dynamic layouts** | ✅ AI adapts to changes | ❌ CSS selectors break |
| **Known structure** | ❌ Slower than direct control | ✅ Fast and precise |
| **Finding elements** | ✅ Natural language queries | ❌ Need exact selectors |
| **Timing control** | ❌ Less predictable | ✅ Full timing control |
| **Complex workflows** | ✅ Handles unexpected UI | ❌ Requires explicit code |

### Browser-Use Best Practices

1. **Start with Agent for exploration**: Let AI navigate complex sites first
2. **Switch to Actor for precision**: Use CSS selectors for predictable elements
3. **Always handle errors**: Websites change—build fallback strategies
4. **Use structured output**: Pydantic models ensure type-safe data
5. **Add delays strategically**: Use `asyncio.sleep()` after actions that trigger changes
6. **Take screenshots**: Visual debugging is invaluable
7. **Combine approaches**: Hybrid workflows leverage strengths of both paradigms

### Real-World Applications

- **Travel Booking**: Monitor prices, auto-book deals, compare options
- **E-commerce**: Track inventory, compare prices, automated purchasing
- **Data Collection**: Scrape dynamic sites, extract structured data
- **Testing**: Automated UI testing with vision-based verification
- **Monitoring**: Check website changes, alert on specific conditions
- **Form Automation**: Fill complex multi-step forms intelligently