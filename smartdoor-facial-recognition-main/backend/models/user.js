const mongoose = require("mongoose");
const loginLogSchema = new mongoose.Schema({
  loginTime: {
    type: String,
  },
  ipAddress: {
    type: String,
    required: true,
  },
  deviceType: {
    type: String,
    required: true,
  },
});

// const faceSchema = new mongoose.Schema({
//     imageData: {
//         type: String,
//         required: true,
//     },
// });

const userSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true,
  },
  password: {
    type: String,
    required: true,
  },
  images: [
    {
      type: Buffer,
      required: true,
    },
  ],
  loginLog: [loginLogSchema],
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

const User = mongoose.model("User", userSchema);
module.exports = User;
