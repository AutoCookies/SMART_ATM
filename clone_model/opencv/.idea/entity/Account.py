import User

class Account(User):
    def __init__(self, user_id, username, fingerprints, face_image, accountNumber, balance=0.0, debt=0.0, bills=None):
        """
        Khởi tạo tài khoản.
        :param user_id: ID người dùng.
        :param username: Tên người dùng.
        :param fingerprints: Dấu vân tay.
        :param face_image: Ảnh khuôn mặt.
        :param accountNumber: Số tài khoản 12 chữ số.
        :param balance: Số dư tài khoản.
        :param debt: Nợ hiện tại.
        :param bills: Danh sách các hóa đơn.
        """
        super().__init__(user_id, username, fingerprints, face_image)
        self.accountNumber = accountNumber
        self.balance = balance
        self.debt = debt
        self.bills = bills if bills is not None else []

    def __str__(self):
        user_info = super().__str__()
        bills_info = "\n".join([str(bill) for bill in self.bills])
        return f"{user_info}, Balance: {self.balance}, Debt: {self.debt}\nBills:\n{bills_info}"