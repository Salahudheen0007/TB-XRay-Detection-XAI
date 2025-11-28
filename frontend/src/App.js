import React, { useState } from "react";
import axios from "axios";
import {
  Container,
  Typography,
  Button,
  Box,
  Card,
  CardContent,
  LinearProgress,
  CircularProgress,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";

const API = "http://127.0.0.1:8000";

// const API = "https://tb-backend-y11d.onrender.com";  // 🔥 YOUR BACKEND URL

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const [gradcamUrl, setGradcamUrl] = useState(null);
  const [limeUrl, setLimeUrl] = useState(null);
  const [xaiLoading, setXaiLoading] = useState(false);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setResult(null);
    setGradcamUrl(null);
    setLimeUrl(null);

    if (f) setPreview(URL.createObjectURL(f));
    else setPreview(null);
  };

  const handlePredict = async () => {
    if (!file) return alert("Please upload an X-ray image first!");

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(
        `${API}/predict`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setResult(res.data);
    } catch (error) {
      console.error(error);
      alert("Prediction failed. Check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  // ---------------------- GRAD CAM ----------------------
  const fetchGradcam = async () => {
    if (!file) return alert("Upload an image first!");

    setXaiLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(
        `${API}/gradcam`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          responseType: "blob",
        }
      );

      const url = URL.createObjectURL(res.data);
      setGradcamUrl(url);
    } catch (err) {
      console.error(err);
      alert("Grad-CAM generation failed.");
    } finally {
      setXaiLoading(false);
    }
  };

  // ---------------------- LIME ----------------------
  const fetchLime = async () => {
    if (!file) return alert("Upload an image first!");

    setXaiLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(
        `${API}/explain`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          responseType: "blob",
        }
      );

      const url = URL.createObjectURL(res.data);
      setLimeUrl(url);
    } catch (err) {
      console.error(err);
      alert("LIME explanation failed.");
    } finally {
      setXaiLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #e8f1ff, #f5f9ff)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px",
      }}
    >
      <Container maxWidth="sm">
        <Card
          elevation={4}
          sx={{
            padding: 4,
            borderRadius: 4,
            backgroundColor: "#ffffffcc",
            backdropFilter: "blur(6px)",
          }}
        >
          <CardContent>
            <Typography
              variant="h4"
              textAlign="center"
              fontWeight="600"
              color="primary"
              gutterBottom
            >
              Tuberculosis Detection
            </Typography>

            <Typography
              textAlign="center"
              color="text.secondary"
              mb={3}
              fontSize={15}
            >
              Upload a chest X-ray image to analyze for Tuberculosis.
            </Typography>

            <Box textAlign="center" mb={2}>
              <Button
                variant="contained"
                component="label"
                startIcon={<CloudUploadIcon />}
                sx={{ paddingX: 3, paddingY: 1, fontWeight: 500, borderRadius: 2 }}
              >
                Upload Image
                <input hidden type="file" accept="image/*" onChange={handleFileChange} />
              </Button>
            </Box>

            {preview && (
              <Box mb={2} textAlign="center">
                <img
                  src={preview}
                  alt="preview"
                  style={{
                    width: "100%",
                    maxHeight: 260,
                    borderRadius: 10,
                    objectFit: "contain",
                    border: "2px solid #e0e6ef",
                  }}
                />
              </Box>
            )}

            <Box textAlign="center">
              <Button
                variant="contained"
                color="primary"
                onClick={handlePredict}
                disabled={loading || !file}
                sx={{ width: "100%", borderRadius: 2, paddingY: 1 }}
              >
                {loading ? "Analyzing..." : "Predict"}
              </Button>
            </Box>

            {loading && (
              <Box mt={3} textAlign="center">
                <CircularProgress color="primary" />
              </Box>
            )}

            {result && (
              <Box mt={3} textAlign="center">
                <Typography
                  variant="h6"
                  fontWeight={600}
                  color={result.prediction === "Tuberculosis" ? "error.main" : "success.main"}
                  display="flex"
                  justifyContent="center"
                  alignItems="center"
                  gap={1}
                >
                  {result.prediction === "Tuberculosis" ? (
                    <ErrorOutlineIcon />
                  ) : (
                    <CheckCircleOutlineIcon />
                  )}
                  {result.prediction}
                </Typography>

                <Box mt={2}>
                  <Typography variant="body2" color="text.secondary">
                    Confidence: {(result.confidence * 100).toFixed(2)}%
                  </Typography>

                  <LinearProgress
                    variant="determinate"
                    value={result.confidence * 100}
                    sx={{ height: 10, borderRadius: 5, mt: 1 }}
                    color={result.prediction === "Tuberculosis" ? "error" : "success"}
                  />
                </Box>
              </Box>
            )}

            {/* ------------------- XAI BUTTONS -------------------- */}
            {result && (
              <Box mt={4} textAlign="center">
                <Typography variant="h6" fontWeight={600} mb={2}>
                  Explainability (XAI)
                </Typography>

                <Button
                  variant="outlined"
                  onClick={fetchGradcam}
                  disabled={xaiLoading}
                  sx={{ width: "100%", mb: 2, borderRadius: 2 }}
                >
                  Generate Grad-CAM
                </Button>

                <Button
                  variant="outlined"
                  onClick={fetchLime}
                  disabled={xaiLoading}
                  sx={{ width: "100%", borderRadius: 2 }}
                >
                  Generate LIME Explanation
                </Button>
              </Box>
            )}

            {xaiLoading && (
              <Box mt={2} textAlign="center">
                <CircularProgress color="secondary" />
              </Box>
            )}

            {/* ------------------- DISPLAY IMAGES -------------------- */}
            {gradcamUrl && (
              <Box mt={4}>
                <Typography textAlign="center" fontWeight={600}>
                  Grad-CAM Heatmap
                </Typography>
                <img
                  src={gradcamUrl}
                  alt="gradcam"
                  style={{
                    width: "100%",
                    borderRadius: 10,
                    marginTop: 10,
                    border: "2px solid #ccc",
                  }}
                />
              </Box>
            )}

            {limeUrl && (
              <Box mt={4}>
                <Typography textAlign="center" fontWeight={600}>
                  LIME Superpixel Explanation
                </Typography>
                <img
                  src={limeUrl}
                  alt="lime"
                  style={{
                    width: "100%",
                    borderRadius: 10,
                    marginTop: 10,
                    border: "2px solid #ccc",
                  }}
                />
              </Box>
            )}
          </CardContent>
        </Card>
      </Container>
    </Box>
  );
}

export default App;
