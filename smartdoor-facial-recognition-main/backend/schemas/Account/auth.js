const joi = require("joi");

const registerUserSchema = joi.object({
  username: joi.string().required().trim().messages({
    "any.required": "Username là bắt buộc",
    "string.empty": "Username không được để trống",
    "string.trim": "Username không được chứa khoảng trắng",
    "string.base": "Username phải là một chuỗi",
  }),

  password: joi.string().min(6).required().messages({
    "any.required": "password là bắt buộc",
    "string.min": "password phải có ít nhất 6 ký tự",
    "string.empty": "password không được để trống",
    "string.base": "password phải là một chuỗi",
  }),
  confirmPassword: joi.string().required().valid(joi.ref("password")).messages({
    "any.required": "confirm password là bắt buộc",
    "any.only": "confirm password không trùng khớp",
    "string.empty": "confirm password không được để trống",
    "string.base": "confirm password phải là một chuỗi",
  }),
});
module.exports = registerUserSchema;
