import React, { useEffect, useReducer, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import styled, { keyframes } from "styled-components";
import Camera from "../component/Camera";
import { loginUser, registerUser } from "../redux/apiRequest";
import { isMobile, isTablet, isBrowser } from "react-device-detect";
const Body = styled.div`
  background-image: url(./Images/background2.jpg);
  background-repeat: no-repeat;
  background-size: cover;
`;
const bounce = keyframes`
  /* 0% { transform: translateX(0); opacity: 0; }
  25% { opacity: 1; }
  50% { transform: translateX(10px); }
  75% { opacity: 1; }
  100% { transform: translateX(0); opacity: 0; } */
  0% {
    opacity: 0;
    transform: translateX(-100%);
  }
  50% {
    opacity: 1;
  }
  100% {
    opacity: 0;
    transform: translateX(100%);
  }
`;
const LoginForm = styled.div`
  background-color: rgba(173, 181, 189, 0.8);
`;
const Text = styled.p`
  /* animation: ${bounce} 2s linear; */
`;
const BtnSubmit = styled.button`
  /* width: 100%; */
  height: 3em;
  border-radius: 8px;
  border: 1px solid #d1d1d1;
  background: rgba(0, 0, 0, 0.75);
  box-shadow: 0px 15px 30px -10px rgba(0, 0, 0, 0.3);
  color: #fff;
  text-align: center;
  font-size: 1em;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  transition: color 0.25s, border-color 0.25s, box-shadow 0.25s, transform 0.25s;
  cursor: pointer;
  padding: 0.5rem 1rem;
  padding-bottom: 2rem;
  align-items: center;
  margin-top: 30px;
  &:hover {
    border-color: rgba(0, 0, 0, 0.75);
    color: white;
    box-shadow: 0 0.5em 0.5em -0.4em rgb(0, 0, 0);
    transform: translate(0, -0.25em);
  }
`;
const initialState = {
  userName: "",
  password: "",
  CFPassword: "",
  isUserNameFocused: false,
  isPasswordFocused: false,
  isCFPasswordFocused: false,
  login: true,
  register: false,
};
const reducer = (state, action) => {
  switch (action.type) {
    case "FOCUS_USERNAME":
      return { ...state, isUserNameFocused: true };
    case "IS_LOGIN":
      return { ...state, login: true, register: false };
    case "IS_REGISTER":
      return { ...state, login: false, register: true };
    case "BLUR_USERNAME":
      return { ...state, isUserNameFocused: false };
    case "FOCUS_PASSWORD":
      return { ...state, isPasswordFocused: true };
    case "BLUR_PASSWORD":
      return { ...state, isPasswordFocused: false };
    case "FOCUS_CFPASSWORD":
      return { ...state, isCFPasswordFocused: true };
    case "BLUR_CFPASSWORD":
      return { ...state, isCFPasswordFocused: false };
    case "CHANGE_USERNAME":
      return { ...state, userName: action.payload };
    case "CHANGE_PASSWORD":
      return { ...state, password: action.payload };
    case "CHANGE_CFPASSWORD":
      return { ...state, CFPassword: action.payload };
    default:
      return state;
  }
};
const Login = () => {
  const navigate = useNavigate();
  const [state, dispatch] = useReducer(reducer, initialState);
  const handleUserNameFocus = () => {
    dispatch({ type: "FOCUS_USERNAME" });
  };

  const handleUserNameBlur = () => {
    dispatch({ type: "BLUR_USERNAME" });
  };

  const handlePasswordFocus = () => {
    dispatch({ type: "FOCUS_PASSWORD" });
  };

  const handlePasswordBlur = () => {
    dispatch({ type: "BLUR_PASSWORD" });
  };
  const handleCFPasswordFocus = () => {
    dispatch({ type: "FOCUS_CFPASSWORD" });
  };

  const handleCFPasswordBlur = () => {
    dispatch({ type: "BLUR_CFPASSWORD" });
  };
  const handleIsLogin = () => {
    dispatch({ type: "IS_LOGIN" });
  };
  const handleIsReGister = () => {
    dispatch({ type: "IS_REGISTER" });
  };
  const handleUserNameChange = (e) => {
    dispatch({ type: "CHANGE_USERNAME", payload: e.target.value });
  };

  const handlePasswordChange = (e) => {
    dispatch({ type: "CHANGE_PASSWORD", payload: e.target.value });
  };
  const handleCFPasswordChange = (e) => {
    dispatch({ type: "CHANGE_CFPASSWORD", payload: e.target.value });
  };
  // console.log(state.login);

  const [images, setImages] = useState([]);
  const [scan, setScan] = useState(false);
  const cameraRef = useRef(null);
  const maxImages = 100;
  const captureCountRef = useRef(0);
  const accessToken = localStorage.getItem("accessToken");
  // const [ip, setIp] = useState("");
  // const [device, setDevice] = useState("");
  const [msg, setMsg] = useState();
  const handleCapture = (blob) => {
    if (state.register && scan) {
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onloadend = () => {
        const binaryData = reader.result;
        setImages((prevImages) => [...prevImages, binaryData]);
        // console.log("Image captured:", binaryData);

        if (captureCountRef.current < maxImages - 1) {
          setTimeout(() => {
            if (cameraRef.current) {
              cameraRef.current.takePicture();
            }
          }, 100);

          captureCountRef.current++;
          // console.log(captureCountRef.current);
        }
      };
    }
  };
  useEffect(() => {
    if (accessToken) {
      navigate("/main");
    }
  }, []);
  useEffect(() => {
    if (state.register && cameraRef.current && scan) {
      setTimeout(() => {
        if (cameraRef.current) {
          captureCountRef.current = 0;
          cameraRef.current.takePicture();
        }
      }, 2000);
    }
  }, [state.register, scan]);
  const handleSubmit = async (e) => {
    try {
      e.preventDefault();
      if (state.register) {
        const newUser = {
          username: state.userName,
          password: state.password,
          confirmPassword: state.CFPassword,
          images: images,
        };
        console.log(newUser);
        const err = await registerUser(newUser, navigate);
        if (err) {
          if (err.message === "Tạo tài khoản thành công") {
            // navigate("/main")
            handleIsLogin();
            const res = await fetch("http://localhost:5050/train_model");
          }
          setMsg(err.message);
        }
      } else {
        if (!state.userName || !state.password) {
          // Hiển thị thông báo lỗi cho người dùng
          // setMsg("Vui lòng cung cấp email và password");
          return;
        }
        // fetch("https://api.ipify.org?format=json")
        //   .then((response) => response.json())
        //   .then((data) => {
        //     setIp(data.ip);
        //   })
        //   .catch((error) => console.error("Error fetching IP:", error));
        const response = await fetch("https://api.ipify.org?format=json");
        const data = await response.json();
        const ipAddress = data.ip;
        // if (isMobile) {
        //   setDevice("Mobile");
        // }
        // if (isTablet) {
        //   setDevice("Tablet");
        // }
        // if (isBrowser) {
        //   setDevice("Browser");
        // }
        let deviceType = "Unknown";
        if (isMobile) {
          deviceType = "Mobile";
        } else if (isTablet) {
          deviceType = "Tablet";
        } else if (isBrowser) {
          deviceType = "Browser";
        }


        const newUser = {
          username: state.userName,
          password: state.password,
          ipAddress: ipAddress,
          deviceType: deviceType,
          loginTime: new Date().toLocaleString("en-US", { timeZone: "Asia/Ho_Chi_Minh" }),
        };

        const res = await loginUser(newUser, navigate);
        if (res) {
          if (res.message === "Đăng nhập thành công") {
            navigate("/main");
          }
          setMsg(res.message);
        }
        console.log(res);
      }
    } catch (err) {
      alert(err);
    }
  };
  return (
    <div>
      <Body className="fixed w-full h-full top-0 left-0 z-[-1] bg-[#a2d2ff]"></Body>
      <div className="flex justify-center items-center w-full h-full my-3  z-0">
        <LoginForm
          onSubmit={handleSubmit}
          className="flex w-full lg:w-1/2 overflow-hidden shadow-xl h-auto justify-center items-center bg-[#edede9]  rounded-lg flex-wrap m-12"
        >
          <Text
            className={`text-lg text-center w-1/2 h-10 cursor-pointer flex items-center justify-center ${state.login ? "bg-[#dedede] animate-pulse" : ""
              }`}
            onClick={handleIsLogin}
          >
            Đăng nhập
          </Text>
          <p
            className={`text-lg text-center w-1/2  h-10 flex items-center justify-center  cursor-pointer ${state.register ? "bg-[#dedede] animate-pulse" : ""
              }`}
            onClick={handleIsReGister}
          >
            Đăng ký
          </p>
          <div className="w-full h-auto min-h-96 m-4 p-4 flex flex-wrap justify-center items-center">
            <form action="" className="w-full flex justify-center flex-wrap">
              <div className="relative w-full mx-10 my-5">
                <input
                  name="userName"
                  id="Input_UserName"
                  onFocus={handleUserNameFocus}
                  onBlur={handleUserNameBlur}
                  onChange={handleUserNameChange}
                  value={state.userName}
                  type="text"
                  className="w-full h-14 py-4 px-3 h-[calc(3.5rem + 2px)] leading-tight border border-[#CDB4DB] block text-base font-normal text-black bg-white bg-clip-padding appearance-none rounded-2xl transition-all pt-7 pb-3 focus-visible:shadow-lg focus-visible:border-red-300 outline-none"
                ></input>
                <label
                  htmlFor="Input_UserName"
                  className={`absolute left-0 top-0 h-full py-4 px-3 pointer-events-none border border-transparent border-solid transition-all text-[#818181] font-normal mb-2 inline-block cursor-default ${state.isUserNameFocused || state.userName
                    ? "opacity-65 scale-75 -translate-y-2 -translate-x-[0.2rem]"
                    : ""
                    }`}
                >
                  User Name
                </label>
              </div>
              <div className="relative w-full mx-10 my-5">
                <input
                  id="Input_Password"
                  type="password"
                  onFocus={handlePasswordFocus}
                  onBlur={handlePasswordBlur}
                  onChange={handlePasswordChange}
                  value={state.password}
                  className="w-full h-14 py-4 px-3 h-[calc(3.5rem + 2px)] leading-tight border border-[#CDB4DB] block text-base font-normal text-black bg-white bg-clip-padding appearance-none rounded-2xl transition-all pt-7 pb-3 focus-visible:shadow-lg focus-visible:border-red-300 outline-none"
                ></input>
                <label
                  htmlFor="Input_Password"
                  className={`absolute left-0 top-0 h-full py-4 px-3 pointer-events-none border border-transparent border-solid transition-all text-[#818181] font-normal mb-2 inline-block cursor-default ${state.isPasswordFocused || state.password
                    ? "opacity-65 scale-75 -translate-y-2 -translate-x-[0.2rem]"
                    : ""
                    }`}
                >
                  Password
                </label>
              </div>
              <div
                className={`relative w-full mx-10 my-5 ${state.register ? "" : "hidden"
                  }`}
              >
                <input
                  id="Input_CFPassword"
                  type="password"
                  onFocus={handleCFPasswordFocus}
                  onBlur={handleCFPasswordBlur}
                  onChange={handleCFPasswordChange}
                  value={state.CFPassword}
                  className="w-full h-14 py-4 px-3 h-[calc(3.5rem + 2px)] leading-tight border border-[#CDB4DB] block text-base font-normal text-black bg-white bg-clip-padding appearance-none rounded-2xl transition-all pt-7 pb-3 focus-visible:shadow-lg focus-visible:border-red-300 outline-none"
                ></input>
                <label
                  htmlFor="Input_CFPassword"
                  className={`absolute left-0 top-0 h-full py-4 px-3 pointer-events-none border border-transparent border-solid transition-all text-[#818181] font-normal mb-2 inline-block cursor-default ${state.isCFPasswordFocused || state.CFPassword
                    ? "opacity-65 scale-75 -translate-y-2 -translate-x-[0.2rem]"
                    : ""
                    }`}
                >
                  Confirm Password
                </label>
              </div>
              <div className="text-center text-red-600 font-semibold w-full text-xl my-6 px-3">
                {msg}
              </div>
              <div
                className={`flex justify-center items-center flex-col w-full h-auto gap-3 ${state.register ? "" : "hidden"
                  }`}
              >
                <div>
                  {captureCountRef.current === 99 ? (
                    <p className="text-green-500">Scan Complete</p>
                  ) : (
                    <p className="text-red-500">
                      Vui lòng giữ khuôn mặt của bạn vào khung hình!
                    </p>
                  )}
                </div>

                <Camera
                  ref={cameraRef}
                  onCapture={handleCapture}
                  video={state.register}
                />
                <div
                  className="cursor-pointer"
                  onClick={() => {
                    setScan(true);
                  }}
                >
                  Scan
                </div>
              </div>

              <div className={`lg:w-2/3 text-center `}>
                <BtnSubmit
                  className={`w-full lg:w-2/3 ${captureCountRef.current === 99
                    ? "cursor: pointer;"
                    : "cursor-not-allowed"
                    }`}
                  type="submit"
                  disabled={
                    state.login ? false : captureCountRef.current !== 99
                  }
                >
                  {state.login ? "Đăng nhập" : "Đăng ký"}
                </BtnSubmit>
              </div>
            </form>
          </div>
        </LoginForm>
      </div>
    </div>
  );
};

export default Login;
