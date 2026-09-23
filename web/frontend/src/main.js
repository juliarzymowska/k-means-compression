const images = ["/hero.JPG", "/hero-4.jpg", "/hero-16.jpg"];

let currentIndex = 0;
let activeLayer = document.getElementById("hero-layer-a");
let inactiveLayer = document.getElementById("hero-layer-b");

function goToImage(direction) {
  const nextIndex = (currentIndex + direction + images.length) % images.length;

  inactiveLayer.style.transition = "none";
  inactiveLayer.style.backgroundImage = `url(${images[nextIndex]})`;
  inactiveLayer.style.transform = `translateX(${direction * 100}%)`;
  inactiveLayer.offsetHeight; // "forced reflow", forcing browser to catch up with style (without it the browser would merge with before style.transistion and skip it entirely)
  inactiveLayer.style.transition = "";

  inactiveLayer.style.transform = "translateX(0%)";
  activeLayer.style.transform = `translateX(${-direction * 100}%)`;

  currentIndex = nextIndex;
  [activeLayer, inactiveLayer] = [inactiveLayer, activeLayer];
}

document
  .getElementById("prev-btn")
  .addEventListener("click", () => goToImage(-1));
document
  .getElementById("next-btn")
  .addEventListener("click", () => goToImage(1));

const resultsSection = document.getElementById("results");

async function handleImageForm(formId, resultId, imgId) {
  const form = document.getElementById(formId);
  const result = document.getElementById(resultId);
  const img = document.getElementById(imgId);
  const submitBtn = form.querySelector('input[type="submit"]');

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const originalLabel = submitBtn.value;
    submitBtn.disabled = true;
    submitBtn.value = "Processing...";

    try {
      const response = await fetch(form.action, {
        method: "POST",
        body: new FormData(form),
      });

      if (!response.ok) {
        alert(await response.text());
        return;
      }

      const blob = await response.blob();
      if (img.src) URL.revokeObjectURL(img.src);
      await new Promise((resolve) => {
        img.addEventListener("load", resolve, { once: true });
        img.src = URL.createObjectURL(blob);
      });
      result.classList.remove("hidden");
      resultsSection.scrollIntoView({ behavior: "smooth" });
    } finally {
      submitBtn.disabled = false;
      submitBtn.value = originalLabel;
    }
  });
}

handleImageForm("compress-form", "compress-result", "compress-output");
handleImageForm("elbow-form", "elbow-result", "elbow-output");
