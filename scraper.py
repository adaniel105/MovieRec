from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time

# Set up the webdriver (make sure you have the appropriate driver installed)
driver = webdriver.Chrome()  # Or use webdriver.Firefox() for Firefox

# Navigate to the IMDb reviews page
url = "https://www.imdb.com/title/tt15314262/reviews"
driver.get(url)

reviews = []

try:
    while True:
        # Wait for review elements to be present
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "div.review-container")
            )
        )

        # Extract reviews
        review_elements = driver.find_elements(By.CSS_SELECTOR, "div.review-container")
        for review in review_elements:
            title = review.find_element(By.CSS_SELECTOR, "a.title").text
            content = review.find_element(By.CSS_SELECTOR, "div.text").text
            rating = review.find_element(
                By.CSS_SELECTOR, "span.rating-other-user-rating"
            ).text

            reviews.append({"title": title, "content": content, "rating": rating})

        # Check if there's a "Load More" button and click it
        try:
            load_more = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button.ipl-load-more__button")
                )
            )
            load_more.click()
            time.sleep(2)  # Wait for new content to load
        except TimeoutException:
            # No more "Load More" button, exit the loop
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the browser
    driver.quit()

filename = "imdb_reviews.txt"
with open(filename, "w", encoding="utf-8") as file:
    for idx, review in enumerate(reviews, 1):
        file.write(f"Rating: {review['rating']}\n")
        file.write(f"Content: {review['content']}...\n\n")

    file.write(f"Total reviews scraped: {len(reviews)}")

print(f"Reviews saved to {filename}")
print(f"Total reviews scraped: {len(reviews)}")
