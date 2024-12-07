const jwt = require("jsonwebtoken");

const middlewareController = {
  //verifyToken
  verifyToken: (req, res, next) => {
    const token = req.headers.token;
    if (!token || typeof token !== "string") {
      return res.status(403).json("Token không hợp lệ");
    }
    if (token) {
      const accessToken = token.split(" ")[1];
      jwt.verify(accessToken, process.env.JWT_ACCESS_KEY, (err, user) => {
        if (err) {
          return res.status(403).json("Token hết hạn");
        }
        req.user = user;
        next();
      });
    } else {
      return res.status(401).json("Bạn chưa đăng nhập");
    }
  },
  //verifyAdmin
  verifyTokenAndAdminAuth: (req, res, next) => {
    middlewareController.verifyToken(req, res, () => {
      if (req.user.role === "admin") {
        next();
      } else {
        return res.status(403).json("Bạn không có quyền admin để làm điều đó");
      }
    });
  },
};
module.exports = middlewareController;
