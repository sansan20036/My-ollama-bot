import axios from "axios";

import { API_BASE } from "../constants/chat";

export const fetchFiles = async () => {
  const response = await axios.get(`${API_BASE}/files`);
  return response.data;
};

export const deleteFile = async (filename) => {
  await axios.delete(`${API_BASE}/files/${encodeURIComponent(filename)}`);
};

export const getFileViewUrl = (filename) =>
  `${API_BASE}/files/${encodeURIComponent(filename)}/view`;

export const fetchBackendStatus = async () => {
  const response = await axios.get(`${API_BASE}/status`);
  return response.data;
};

export const fetchModels = async () => {
  const response = await axios.get(`${API_BASE}/models`);
  return response.data;
};

export const uploadFiles = async (formData) => {
  const response = await axios.post(`${API_BASE}/upload`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const resetSystem = async () => {
  await axios.post(`${API_BASE}/reset`);
};

export const sendChatStream = ({
  query,
  modelName,
  history,
  images,
  signal,
}) =>
  fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      model_name: modelName,
      history,
      images,
    }),
    signal,
  });
