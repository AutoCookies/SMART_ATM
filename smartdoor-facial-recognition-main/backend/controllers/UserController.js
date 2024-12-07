const { json } = require("body-parser");
const user = require("../models/user");

const getAllUser = async (req, res) => {
    try {
        const data = await user.find().select('-images');
        if (data.length < 0) {
            return res.status(404).json({ message: "No user found " });
        }
        res.status(201).json(data);
    } catch (error) {
        return res.status(404).json({
            message: error.message,
        });
    }
};
module.exports = {
    getAllUser
};