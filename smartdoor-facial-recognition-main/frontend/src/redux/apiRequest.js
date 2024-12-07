import axios from "axios";

export const registerUser = async (user, navigate) => {
  // dispatch(registerStart());
  try {
    const err = await axios.post(
      "http://localhost:5000/api/auth/register",
      user
    );
    // navigate("/main");
    if (err.data) {
      return err.data;
    }
    return err;
    //   dispatch(registerSuccess());
  } catch (error) {
    //   dispatch(registerFalse(error.response.data));
    return error.response?.data;
  }
};
export const loginUser = async (user, navigate) => {
  // dispatch(loginStart());
  try {
    const res = await axios.post("http://localhost:5000/api/auth/login", user, {
      withCredentials: true,
    });
    // dispatch(loginSuccess(res.data));
    if (res.data) {
      localStorage.setItem("accessToken", res.data.accessToken);
      return res.data;
    }
    return res;
  } catch (error) {
    // dispatch(loginFalse(error.response.data));
    return error.response?.data;
  }
};
