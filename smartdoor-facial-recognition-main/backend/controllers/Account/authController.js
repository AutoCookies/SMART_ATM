const User = require("../../models/user");
const bcrypt = require("bcrypt");
const registerUserSchema = require("../../schemas/Account/auth");
const jwt = require("jsonwebtoken");
let refreshTokens = [];
const authController = {
  registerUser: async (req, res) => {
    try {
      const salt = await bcrypt.genSalt(10);
      const { username, password, confirmPassword, images } = req.body;

      // Validate data
      const { error } = registerUserSchema.validate(
        { username, password, confirmPassword },
        {
          abortEarly: false,
        }
      );

      if (error) {
        const messages = error.details.map((message) => message.message);
        return res.status(400).json({
          message: messages,
        });
      }

      // Check if user exists
      const existUser = await User.findOne({
        username,
      });

      if (existUser) {
        return res.status(400).json({
          message: ["username đã được sử dụng"],
        });
      }

      // Hash password
      const hashed = await bcrypt.hash(password, salt);

      // Create new user
      const newUser = new User({
        username,
        password: hashed,
        images: images,
      });
      const user = await newUser.save();
      return res
        .status(200)
        .json({ user, message: "Tạo tài khoản thành công" });
    } catch (err) {
      return res.status(500).json(err);
    }
  },
  generateAccessToken: (user) => {
    // console.log(user._id);
    return jwt.sign(
      {
        id: user._id,
        username: user.username,

        // role: user.role,
      },
      process.env.JWT_ACCESS_KEY,
      { expiresIn: "30s" }
    );
  },
  generateRefreshToken: (user) => {
    // console.log(user._id);
    return jwt.sign(
      {
        id: user._id,
        username: user.username,
        // role: user.role,
      },
      process.env.JWT_REFRESH_KEY,
      { expiresIn: "365d" }
    );
  },
  loginUser: async (req, res) => {
    try {
      const { username, password, loginTime, ipAddress, deviceType } = req.body;
      if (!username || !password) {
        return res
          .status(400)
          .json({ message: "Vui lòng nhập username hoặc password!" });
      }
      const user = await User.findOne({ username });
      if (!user) {
        return res.status(400).json({ message: "Sai tên đăng nhập!" });
      }
      const validPassword = await bcrypt.compare(password, user.password);
      if (!validPassword) {
        return res.status(400).json({ message: "Sai mật khẩu!" });
      }
      if (user && validPassword) {
        const accessToken = authController.generateAccessToken(user);
        const refreshToken = authController.generateRefreshToken(user);
        refreshTokens.push(refreshToken);
        const oneYearInSeconds = 365 * 24 * 60 * 60;
        // console.log("meo meo");
        const expireDate = new Date();
        const loginLog = { loginTime, ipAddress, deviceType };
        user.loginLog.push(loginLog);
        await user.save();
        // console.log(user.loginLog);
        // console.log("meo meo");
        expireDate.setTime(expireDate.getTime() + oneYearInSeconds * 1000);
        // console.log("meo meo");
        res.cookie("refreshToken", refreshToken, {
          httpOnly: true,
          secure: false,
          path: "/",
          sameSite: "strict",
          expires: expireDate,
        });

        const { password, images, ...other } = user._doc;
        // console.log("12 meo");
        // console.log(...other);
        // console.log("12 meo");
        return res
          .status(200)
          .json({ ...other, accessToken, message: "Đăng nhập thành công" });
      }
    } catch (err) {
      return res.status(500).json(err);
    }
  },
  requestRefreshToken: async (req, res) => {
    //Take refresh token from userf
    const refreshToken = req.cookies.refreshToken;
    if (!refreshToken) return res.status(401).json("You're not authenticated");
    // if (!refreshTokens.includes(refreshToken)) {
    //   return res.status(403).json("Refresh token is not valid");
    // }
    jwt.verify(refreshToken, process.env.JWT_REFRESH_KEY, (err, user) => {
      if (err) {
        return res.status(401).json(err);
      }
      refreshTokens = refreshTokens.filter((token) => token !== refreshToken);
      const newAccessToken = authController.generateAccessToken(user);
      const newRefeshToken = authController.generateRefreshToken(user);
      refreshTokens.push(newRefeshToken);
      const oneYearInSeconds = 365 * 24 * 60 * 60;
      const expireDate = new Date();
      expireDate.setTime(expireDate.getTime() + oneYearInSeconds * 1000);
      res.cookie("refreshToken", newRefeshToken, {
        httpOnly: true,
        secure: false,
        path: "/",
        sameSite: "strict",
        expires: expireDate,
      });
      return res.status(200).json({ accessToken: newAccessToken });
    });
  },
  userLogout: async (req, res) => {
    res.clearCookie("refreshToken");
    refreshTokens = refreshTokens.filter(
      (token) => token !== req.cookies.refreshToken
    );
    return res.status(200).json("Đăng xuất thành công!");
  },
  verifyToken: (token) => {
    try {
      return jwt.verify(token, process.env.JWT_ACCESS_KEY);
    } catch (err) {
      return null;
    }
  },
};

module.exports = authController;
