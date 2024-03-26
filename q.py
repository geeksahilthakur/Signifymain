# Importing the necessary library for generating QR codes
import pyqrcode


# Function to generate UPI QR code
def generate_upi_qr_code(upi_id, user_name):
    # Constructing the UPI URL using the provided information
    upi_url = f"upi://pay?pa={upi_id}&pn={user_name}&cu=INR"

    # Generating QR code
    qr_code = pyqrcode.create(upi_url)  # Generating QR code directly with the URL

    return qr_code  # Returning the QR code object


if __name__ == "__main__":
    # Prompting the user to input necessary details
    upi_id = input("Enter UPI ID: ")
    user_name = input("Enter User Name: ")

    # Generating QR code using the provided information
    qr_code = generate_upi_qr_code(upi_id, user_name)

    # Saving the QR code as a PNG file named "qr_code.png"
    qr_code.png("qr_code.png", scale=10)
    print("QR code saved as 'qr_code.png'.")
