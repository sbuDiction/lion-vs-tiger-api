import handlebars from 'vite-plugin-handlebars';
import { defineConfig } from 'vite';

export default defineConfig({
    base: '/wrp/',
    build: {
        rollupOptions: {
            input: {
                main: 'index.html',
            }
        },
    },
    plugins: [handlebars({
        context: {
            // some data here.
        }
    })]

});