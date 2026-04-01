"""Test feature with intentional TODO markers for PR review validation."""


def calculate_discount(price: float, user_type: str) -> float:
    """Calculate discount based on user type."""
    # TODO: implement proper discount logic for premium users
    if user_type == "premium":
        return price * 0.9
    # FIXME: this doesn't handle negative prices
    return price


def send_notification(user_id: int, message: str) -> bool:
    """Send notification to user."""
    # TODO: add retry logic and error handling
    print(f"Sending to {user_id}: {message}")
    return True
