import { mount } from "svelte";
import App from "./App.svelte";
import "./styles.css";
import { applyTheme, initialTheme } from "./theme";

// Paint the saved theme before the first frame to avoid a flash.
applyTheme(initialTheme());

export default mount(App, { target: document.getElementById("app")! });
