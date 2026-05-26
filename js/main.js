// ===== Hamburger Menu =====
(function () {
    var btn = document.getElementById('hamburger');
    var links = document.getElementById('navLinks');
    if (!btn || !links) return;
    btn.addEventListener('click', function () {
        btn.classList.toggle('active');
        links.classList.toggle('show');
    });
})();

// ===== Article Expand/Collapse =====
(function () {
    var cards = document.querySelectorAll('.article-card');
    cards.forEach(function (card) {
        card.addEventListener('click', function () {
            var wasExpanded = card.classList.contains('expanded');
            // Close all in the same list
            var list = card.parentElement;
            if (list) {
                list.querySelectorAll('.article-card.expanded').forEach(function (c) {
                    c.classList.remove('expanded');
                });
            }
            if (!wasExpanded) {
                card.classList.add('expanded');
            }
        });
    });
})();

// ===== Back to Top =====
(function () {
    var btn = document.getElementById('backToTop');
    if (!btn) return;
    var ticking = false;
    window.addEventListener('scroll', function () {
        if (!ticking) {
            requestAnimationFrame(function () {
                if (window.scrollY > 400) {
                    btn.classList.add('show');
                } else {
                    btn.classList.remove('show');
                }
                ticking = false;
            });
            ticking = true;
        }
    });
    btn.addEventListener('click', function () {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
})();

// ===== Highlight active nav link =====
(function () {
    var current = window.location.pathname.split('/').pop() || 'index.html';
    var links = document.querySelectorAll('.nav-links a');
    links.forEach(function (a) {
        if (a.getAttribute('href') === current) {
            a.classList.add('active');
        }
    });
})();
