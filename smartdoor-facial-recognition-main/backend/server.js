const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const bodyParser = require("body-parser");
require("dotenv").config();
const authRoute = require("./routes/account/authRoute");
const userRoute = require("./routes/userRoute"); // Import route người dùng
const app = express();
const port = process.env.PORT || 5000;

app.use(
  cors({
    origin: "http://localhost:3000",
    credentials: true,
  })
);
app.use(bodyParser.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ limit: "50mb", extended: true }));
app.use(express.json());

app.use("/api/auth", authRoute);
app.use("/api", userRoute); // Sử dụng route người dùng

mongoose
  .connect(process.env.DB_URI)
  .then(() => console.log("Kết nối cơ sở dữ liệu thành công"))
  .catch((err) => console.error("Lỗi kết nối cơ sở dữ liệu", err));

app.get("/", (req, res) => {
  res.send("Xin chào");
});

app.listen(port, () => {
  console.log(`Server chạy trên cổng: ${port}`);
});
