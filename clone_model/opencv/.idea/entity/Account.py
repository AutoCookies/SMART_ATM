import User
class Account(User):
    def __init__(self, user_id, username, fingerprint, face_image, balance=0.0, debt=0.0):
        super().__init__(user_id, username, fingerprint, face_image)
        self.balance = balance
        self.debt = debt

    def deposit(self, amount):
        """Nạp tiền vào tài khoản."""
        if amount <= 0:
            raise ValueError("Số tiền nạp phải lớn hơn 0.")
        self.balance += amount
        print(f"Nạp thành công {amount}. Số dư hiện tại: {self.balance}")

    def withdraw(self, amount):
        """Rút tiền từ tài khoản."""
        if amount <= 0:
            raise ValueError("Số tiền rút phải lớn hơn 0.")
        if amount > self.balance:
            raise ValueError("Không đủ số dư trong tài khoản.")
        self.balance -= amount
        print(f"Rút thành công {amount}. Số dư còn lại: {self.balance}")

    def repay_debt(self, amount):
        """Thanh toán nợ."""
        if amount <= 0:
            raise ValueError("Số tiền thanh toán phải lớn hơn 0.")
        if amount > self.debt:
            raise ValueError("Số tiền thanh toán vượt quá số nợ.")
        self.debt -= amount
        print(f"Thanh toán thành công {amount}. Nợ còn lại: {self.debt}")

    def __str__(self):
        user_info = super().__str__()
        return f"{user_info}, Balance: {self.balance}, Debt: {self.debt}"
