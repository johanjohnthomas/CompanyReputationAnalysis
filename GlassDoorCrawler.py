import os
import time
import logging
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def init_driver():
    """
    Initializes a Chrome driver for Selenium with custom headers.
    This version runs in non-headless mode so that you can interact with the browser.
    """
    chrome_options = Options()
    # Do not run headless so the browser window is visible
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    # Set a custom User-Agent to mimic a regular browser
    custom_user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
    chrome_options.add_argument(f'--user-agent={custom_user_agent}')
    chrome_options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def get_reviews(driver, company, url):
    """
    Loads the Glassdoor reviews page for the specified company,
    pauses for manual captcha solving, and extracts review data.
    
    Args:
        driver: Selenium webdriver instance.
        company: String, the company name.
        url: String, URL of the Glassdoor reviews page.
        
    Returns:
        A list of dictionaries, each representing a single review.
    """
    reviews = []
    logger.info("Fetching reviews for %s from %s", company, url)
    driver.get(url)
    
    # Pause execution and prompt the user to solve the captcha manually.
    input("If a captcha appears, please solve it in the opened browser, then press Enter to continue...")

    # Wait for review elements to load
    wait = WebDriverWait(driver, 15)
    try:
        # Using an updated selector: Glassdoor often uses article elements with data-test attributes.
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "article[data-test='emp-review']")))
    except Exception as e:
        logger.error("Reviews did not load in time: %s", e)
        return reviews

    # Attempt to dismiss any lingering cookie consent (if present)
    try:
        consent_button = driver.find_element(By.CSS_SELECTOR, "button#onetrust-accept-btn-handler")
        if consent_button:
            consent_button.click()
            time.sleep(2)
    except Exception as e:
        logger.info("No cookie consent button found or error: %s", e)

    # Scroll down to ensure all dynamic content loads
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    review_containers = soup.find_all("article", {"data-test": "emp-review"})
    if not review_containers:
        review_containers = soup.find_all("li", class_="empReview")
    logger.info("Found %d review containers", len(review_containers))

    for container in review_containers:
        try:
            title_tag = container.find("a", {"data-test": "reviewLink"})
            review_title = title_tag.get_text(strip=True) if title_tag else ""
            date_tag = container.find("time")
            review_date = date_tag.get_text(strip=True) if date_tag else ""
            rating_tag = container.find("span", {"data-test": "rating"})
            rating = rating_tag.get_text(strip=True) if rating_tag else ""
            pros_tag = container.find("span", {"data-test": "pros"})
            cons_tag = container.find("span", {"data-test": "cons"})
            review_text = ""
            if pros_tag:
                review_text += "Pros: " + pros_tag.get_text(strip=True) + " "
            if cons_tag:
                review_text += "Cons: " + cons_tag.get_text(strip=True)
            emp_info_tag = container.find("span", {"data-test": "authorJobTitle"})
            employee_role = emp_info_tag.get_text(strip=True) if emp_info_tag else ""
            reviews.append({
                "company": company,
                "review_title": review_title,
                "review_date": review_date,
                "rating": rating,
                "review_text": review_text,
                "employee_role": employee_role
            })
        except Exception as ex:
            logger.error("Error parsing review container: %s", ex)
    return reviews

def main():
    driver = init_driver()
    
    # Dictionary of companies and their Glassdoor review page URLs.
    companies = {
        "Samsung": "https://www.glassdoor.com/Reviews/Samsung-Electronics-Reviews-EI_IE7588.0,17.htm",
        "Apple": "https://www.glassdoor.com/Reviews/Apple-Reviews-EI_IE1138.0,5.htm"
    }
    
    all_reviews = []
    for company, url in companies.items():
        company_reviews = get_reviews(driver, company, url)
        all_reviews.extend(company_reviews)
    
    driver.quit()
    
    # Save results to CSV
    df = pd.DataFrame(all_reviews)
    output_file = "employee_reviews.csv"
    df.to_csv(output_file, index=False)
    logger.info("Saved %d reviews to %s", len(df), output_file)

if __name__ == "__main__":
    main()
