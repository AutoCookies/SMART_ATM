import React, { useState, useRef, useEffect } from "react";
import Camera from "../component/Camera";
import { jwtDecode } from "jwt-decode";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";

const Main = () => {
  const [imageData, setImageData] = useState(null);
  const cameraRef = useRef(null);
  const accessToken = localStorage.getItem("accessToken");
  const [decoded, setDecoded] = useState(null);
  const [label, setLabel] = useState("");
  const navigate = useNavigate();
  useEffect(() => {
    // Kiểm tra nếu không có accessToken
    if (!accessToken) {
      // Điều hướng về trang đăng nhập
      navigate("/");
    } else {
      // Nếu có accessToken thì mới decode nó
      try {
        const decodedToken = jwtDecode(accessToken);
        setDecoded(decodedToken); // Lưu giá trị decoded vào state
        // console.log(decodedToken); // Hoặc thực hiện các hành động khác với dữ liệu decoded
      } catch (error) {
        console.error("Token không hợp lệ:", error);
        // Nếu token không hợp lệ, điều hướng về trang đăng nhập
        navigate("/");
      }
    }
  }, [accessToken, navigate]);
  const handleLogout = () => {
    localStorage.clear();
    navigate("/");
  };

  const handleCapture = (blob) => {
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.onloadend = () => {
      const binaryData = reader.result;
      setImageData(binaryData);
      console.log("Binary Data:", binaryData);
    };
  };

  const handleSaveImage = () => {
    if (cameraRef.current) {
      cameraRef.current.takePicture();
    }
  };

  const handleRecognition = async () => {
    if (!imageData) {
      console.error("No image data to send.");
      return;
    }

    try {
      const response = await axios.post(
        "http://localhost:5050/recognize",
        {
          image: imageData, // imageData là chuỗi Base64
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      setLabel(response.data.label);
      console.log("Recognized label:", response.data.label);
    } catch (error) {
      console.error(
        "Error:",
        error.response ? error.response.data.message : error.message
      );
    }
  };
  console.log(decoded);
  return (
    <div>
      <nav className="flex lg:items-center lg:justify-between lg:flex-row flex-col w-full">
        <h2 className="items-start lg:mx-auto font-bold text-2xl font-serif my-4 mx-2">
          Smart Door Facial Recognition
        </h2>
        <div className="flex justify-end items-center gap-4 lg:m-8 mx-2 mb-20">
          <Link to="/users" className="text-md hover:bg-slate-300 px-4 py-2 rounded-lg font-semibold">Quản Lý Users</Link>
          <button
            className="text-md hover:bg-slate-300 px-4 py-2 rounded-lg font-semibold"
            onClick={handleRecognition}
          >
            Nhận Diện Khuôn Mặt
          </button>
          <button className="text-md font-semibold cursor-default">
            Hello, {decoded ? decoded.username : ""}
          </button>
          <button
            onClick={handleLogout}
            className="px-6 py-3 bg-black rounded-3xl text-md font-semibold text-white cursor-pointer hover:opacity-70 transition-opacity"
          >
            Log out
          </button>
        </div>
      </nav>
      <div className="flex justify-center items-center flex-col gap-6">
        <p className="text-lg">{label}</p>
        <Camera ref={cameraRef} onCapture={handleCapture} video={true} />
        <button
          className="px-4 py-2 text-lg bg-green-400 hover:bg-green-500 rounded-md text-white "
          onClick={handleSaveImage}
        >
          Chụp Hình
        </button>
      </div>
    </div>
  );
};

export default Main;
