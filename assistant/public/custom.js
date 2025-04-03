// Global Constants
const appName = "PTX Assistant";

// Messages
const footerMsg = ["made with", "❤️", "by", "TeamPTXBuddy"];
const disclaimer = "Verify AI response for accuracy";

// Node Selectors
const classAIMessage = "ai-message";
const classMsgElementsLight = "css-1d7zxcn";
const classMsgElementsDark = "css-gnlfwc";
const classFooter = "watermark";

// Configuration objects
const mutationObserverConfig = {
  childList: true, // Observe direct children additions/removals
  subtree: true, // Observe all descendants
  attributes: false, // Don't observe attribute changes
};

// Common Styles
const fontColor = "#666666";
const heartColor = "#e53935";

// Helper functions for creating elements
function createInfoSpan() {
  const span = document.createElement("span");
  span.style.fontSize = "xx-small";
  span.innerHTML = "&nbsp;&nbsp;" + disclaimer;
  span.classList.add("info-span-marker"); // Add a class to mark spans we've added
  return span;
}

function createFooterContent() {
  const p = document.createElement("p");
  p.style.margin = "0px";
  p.style.color = fontColor;

  p.classList.add("replaced-footer");

  // Create "made with" span
  const madeWithSpan = document.createElement("span");
  madeWithSpan.textContent = footerMsg[0];
  p.appendChild(madeWithSpan);

  // Add space
  p.appendChild(document.createTextNode("\u00A0")); // &nbsp;

  // Create heart span
  const heartSpan = document.createElement("span");
  heartSpan.style.color = heartColor;
  heartSpan.textContent = footerMsg[1];
  p.appendChild(heartSpan);

  // Add space
  p.appendChild(document.createTextNode("\u00A0")); // &nbsp;

  // Create "by" span
  const bySpan = document.createElement("span");
  bySpan.textContent = footerMsg[2];
  p.appendChild(bySpan);

  // Add space
  p.appendChild(document.createTextNode("\u00A0")); // &nbsp;

  // Create author span
  const authorSpan = document.createElement("span");
  authorSpan.style.fontWeight = "bolder";
  authorSpan.textContent = footerMsg[3];
  p.appendChild(authorSpan);

  return p;
}

// Message processing functions
function processAIMessageDiv(div) {
  const msgElementsDiv =
    div.querySelector(`.${classMsgElementsLight}`) ||
    div.querySelector(`.${classMsgElementsDark}`);
  if (msgElementsDiv) {
    // Find the avatar container and check its aria-label
    const avatarContainer = div.querySelector(
      ".message-avatar div[aria-label]"
    );
    if (
      !avatarContainer ||
      avatarContainer.getAttribute("aria-label") !== appName
    ) {
      msgElementsDiv.remove();
      console.log("Removed message elements from the AI generated message!");
      return;
    } else if (!msgElementsDiv.querySelector(".info-span-marker")) {
      // Add disclaimer to the AI generated message
      const span = createInfoSpan();
      msgElementsDiv.appendChild(span);
      console.log("Added disclaimer to the AI generated message!");
    }
  }
}

// Footer processing functions
function replaceFooterContent(footer) {
  // Check if footer already contains a paragraph with class "replaced-footer"
  if (footer.querySelector("p.replaced-footer")) {
    return false; // Footer already contains the correct content
  }

  // Clear existing content
  footer.innerHTML = "";
  // Create and append new content
  footer.appendChild(createFooterContent());
  return true;
}

function handleFooterUpdate(footer) {
  const isReplaced = replaceFooterContent(footer);
  if (isReplaced) {
    console.log("Updated the Footer!");
  }
}

// Mutation handling functions
function handleMutations(mutations) {
  mutations.forEach((mutation) => {
    // Handle messages
    mutation.addedNodes.forEach((node) => {
      if (node.nodeType === 1) {
        // Check for message elements div
        if (
          node.classList.contains(classMsgElementsLight) ||
          node.classList.contains(classMsgElementsDark)
        ) {
          processAIMessageDiv(node.parentNode.parentNode);
        }

        // Check for AI message divs
        const aiMessage = node.querySelector(`.${classAIMessage}`);
        if (aiMessage) {
          processAIMessageDiv(aiMessage);
        }

        // Check for footer
        const footer = node.querySelector(`.${classFooter}`);
        if (footer) {
          handleFooterUpdate(footer);
        }
      }
    });
  });
}

// Observer initialization and management
function initializeObserver() {
  try {
    // Create the observer instance
    const observer = new MutationObserver(handleMutations);

    // Start observing the target (document.body for full page coverage)
    observer.observe(document.body, mutationObserverConfig);

    // Process existing footer if present
    const existingFooter = document.querySelector(`.${classFooter}`);
    if (existingFooter) {
      handleFooterUpdate(existingFooter);
    }

    // Return the observer in case we need to disconnect it later
    return observer;
  } catch (error) {
    console.error("Error initializing observer:", error);
    return null;
  }
}

// Cleanup function
function cleanup(observer) {
  if (observer) {
    observer.disconnect();
  }
}

// Initialization logic
function initialize() {
  const observer = initializeObserver();

  // Cleanup on page unload
  window.addEventListener("unload", () => cleanup(observer));

  return observer;
}

// Initialize the script
initialize();
