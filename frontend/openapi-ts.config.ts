import { defaultPlugins, defineConfig } from "@hey-api/openapi-ts";

// Generates the typed client from the backend's exported schema.
// Run `npm run gen:api` after changing any Pydantic model (it's also a predev hook).
export default defineConfig({
  input: "./openapi.json",
  output: {
    path: "./src/api/generated",
  },
  plugins: [...defaultPlugins, "@hey-api/client-fetch"],
});
