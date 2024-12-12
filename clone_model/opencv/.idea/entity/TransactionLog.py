class TransactionLog:
    def __init__(self, transaction_id, transaction_type, account_number, amount, date, note):
        """
        Ghi nhận lịch sử giao dịch.
        :param transaction_id: ID giao dịch.
        :param transaction_type: Loại giao dịch (Deposit, Withdrawal, Transfer).
        :param account_number: Số tài khoản liên quan.
        :param amount: Số tiền.
        :param date: Ngày giao dịch.
        :param note: Ghi chú giao dịch.
        """
        self.transaction_id = transaction_id
        self.transaction_type = transaction_type
        self.account_number = account_number
        self.amount = amount
        self.date = date
        self.note = note

    def __str__(self):
        return (
            f"Transaction ID: {self.transaction_id}, "
            f"Type: {self.transaction_type}, "
            f"Account: {self.account_number}, "
            f"Amount: {self.amount}, "
            f"Date: {self.date}, "
            f"Note: {self.note}"
        )