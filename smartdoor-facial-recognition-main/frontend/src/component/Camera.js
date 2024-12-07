import React, {
  useRef,
  useEffect,
  forwardRef,
  useImperativeHandle,
} from "react";

const Camera = forwardRef((props, ref) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const { video } = props;
  // console.log("aa: " + video)
  useEffect(() => {
    const openCamera = async () => {
      if (!video) {
        // Nếu video là false, không mở camera
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera: ", err);
      }
    };

    openCamera();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject;
        const tracks = stream.getTracks();

        tracks.forEach((track) => {
          track.stop();
        });
      }
    };
  }, [video]);

  useImperativeHandle(ref, () => ({
    takePicture() {
      const canvas = canvasRef.current;
      const video = videoRef.current;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob((blob) => {
        props.onCapture(blob);
      }, "image/jpeg");
    },
  }));

  return (
    <div className="flex justify-center overflow-hidden w-full">
      <video
        className="lg:h-[400px] lg:w-[400px] h-[300px] w-[300px] rounded-[50%]"
        ref={videoRef}
        autoPlay
        style={{ objectFit: "cover" }}
      ></video>
      <canvas ref={canvasRef} style={{ display: "none" }}></canvas>
    </div>
  );
});

export default Camera;
