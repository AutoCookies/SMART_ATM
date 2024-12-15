import User
import TransactionLog
from datetime import datetime

class Account(User):
    def __init__(self, user_id, username, fingerprints, face_image, accountNumber, balance=0.0, debt=0.0, bills=None):
        """
        Khởi tạo tài khoản.
        """
        super().__init__(user_id, username, fingerprints, face_image)
        self.accountNumber = accountNumber
        self.balance = balance
        self.debt = debt
        self.bills = bills if bills is not None else []
        self.transaction_logs = []  # Ghi nhận lịch sử giao dịch

    def log_transaction(self, transaction_type, amount, note):
        transaction_id = len(self.transaction_logs) + 1
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = TransactionLog(transaction_id, transaction_type, self.accountNumber, amount, date, note)
        self.transaction_logs.append(log)

    def __str__(self):
        user_info = super().__str__()
        bills_info = "\n".join([str(bill) for bill in self.bills])
        transactions = "\n".join([str(t) for t in self.transaction_logs])
        return f"{user_info}, Balance: {self.balance}, Debt: {self.debt}\nBills:\n{bills_info}\nTransaction Logs:\n{transactions}"