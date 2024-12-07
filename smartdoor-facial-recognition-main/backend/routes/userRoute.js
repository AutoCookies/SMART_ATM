const express = require("express");
const User = require("../models/user"); // Import model người dùng
const router = express.Router();
const UserController = require("../controllers/UserController");
// Route để lấy danh sách người dùng từ MongoDB
router.get("/users", UserController.getAllUser);
module.exports = router;
