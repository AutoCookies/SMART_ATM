class Bills:
    def __init__(self, bill_id, bill_type, start_date, deadline, total_amount, paid_amount=0.0, is_paid=False):
        """
        Khởi tạo hóa đơn.
        :param bill_id: ID của hóa đơn.
        :param bill_type: Loại hóa đơn (ví dụ: điện, nước, internet, ...).
        :param start_date: Ngày phát hành hóa đơn.
        :param deadline: Hạn thanh toán.
        :param total_amount: Tổng số tiền cần thanh toán.
        :param paid_amount: Số tiền đã thanh toán (mặc định là 0.0).
        :param is_paid: Trạng thái thanh toán (True nếu đã thanh toán, False nếu chưa).
        """
        self.bill_id = bill_id
        self.bill_type = bill_type
        self.start_date = start_date
        self.deadline = deadline
        self.total_amount = total_amount
        self.paid_amount = paid_amount
        self.is_paid = is_paid

    def __str__(self):
        """Định dạng thông tin hóa đơn dưới dạng chuỗi."""
        return (
            f"Bill ID: {self.bill_id}, "
            f"Type: {self.bill_type}, "
            f"Total: {self.total_amount}, "
            f"Paid: {self.paid_amount}, "
            f"Is Paid: {self.is_paid}"
        )
