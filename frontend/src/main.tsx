import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";
import { client } from "./api/generated/client.gen";
import { App } from "./App";
import "./styles.css";

// Same-origin in dev: Vite proxies /v1 + /health to the backend (see vite.config.ts).
client.setConfig({ baseUrl: "" });

const queryClient = new QueryClient();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
