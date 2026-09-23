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
