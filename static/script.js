const mobileMenu = document.getElementById('mobile-menu');
const navList = document.querySelector('.nav-list');

mobileMenu.addEventListener('click', () => {
    navList.classList.toggle('show'); // Adiciona ou remove a classe 'show' ao clicar
});



