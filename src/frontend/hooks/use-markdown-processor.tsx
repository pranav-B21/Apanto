import * as React from "react";
import { Dialog, DialogContent } from "../components/ui/dialog";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import "highlight.js/styles/github-dark.css";
import mermaid from "mermaid";

export const useMarkdownProcessor = (content: string): React.ReactNode => {
  React.useEffect(() => {
    mermaid.initialize({ 
      startOnLoad: false, 
      theme: "dark",
      darkMode: true,
      securityLevel: 'loose'
    });
  }, []);

  return React.useMemo(() => {
    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          code: CodeBlock,
          a: ({ node, href, children }) => (
            <a href={href} target="_blank" rel="noreferrer" className="text-blue-600 hover:text-blue-800 underline">
              {children}
            </a>
          ),
          h1: ({ node, children }) => <h1 className="text-2xl font-bold mb-4">{children}</h1>,
          h2: ({ node, children }) => <h2 className="text-xl font-bold mb-3">{children}</h2>,
          h3: ({ node, children }) => <h3 className="text-lg font-bold mb-2">{children}</h3>,
          h4: ({ node, children }) => <h4 className="text-md font-bold mb-2">{children}</h4>,
          h5: ({ node, children }) => <h5 className="text-sm font-bold mb-2">{children}</h5>,
          h6: ({ node, children }) => <h6 className="text-xs font-bold mb-2">{children}</h6>,
          p: ({ node, children }) => <p className="mb-2 last:mb-0">{children}</p>,
          ul: ({ node, children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
          ol: ({ node, children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
          li: ({ node, children }) => <li className="ml-4">{children}</li>,
          table: ({ node, children }) => (
            <div className="overflow-x-auto my-4">
              <table className="min-w-full border-collapse border border-gray-300">{children}</table>
            </div>
          ),
          thead: ({ node, children }) => <thead className="bg-gray-100">{children}</thead>,
          th: ({ node, children }) => <th className="border border-gray-300 px-4 py-2 font-bold">{children}</th>,
          td: ({ node, children }) => <td className="border border-gray-300 px-4 py-2">{children}</td>,
          blockquote: ({ node, children }) => (
            <blockquote className="border-l-4 border-gray-300 pl-4 italic my-2">{children}</blockquote>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    );
  }, [content]);
};

const CodeBlock: React.FC<any> = ({ node, inline, className, children }) => {
  const [showMermaidPreview, setShowMermaidPreview] = React.useState(false);
  const [diagram, setDiagram] = React.useState<string | null>(null);

  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';
  const isMermaid = language === 'mermaid';

  React.useEffect(() => {
    if (isMermaid && children) {
      const render = async () => {
        const id = `mermaid-${Math.round(Math.random() * 10000000)}`;
        try {
          if (await mermaid.parse(children.toString(), { suppressErrors: true })) {
            const { svg } = await mermaid.render(id, children.toString());
            setDiagram(svg);
          }
        } catch (error) {
          console.error('Failed to render Mermaid diagram:', error);
          setDiagram(null);
        }
      };
      render();
    }
  }, [isMermaid, children]);

  if (inline) {
    return <code className="px-1.5 py-0.5 bg-gray-100 rounded text-sm font-mono">{children}</code>;
  }

  return (
    <div>
      <div className="relative my-4 rounded-lg overflow-hidden">
        <pre className="p-4 bg-gray-800 text-gray-100 overflow-x-auto">
          <code className={className}>{children}</code>
        </pre>
      </div>
      {isMermaid && (
        <div className="mt-2">
          <button
            type="button"
            className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
            onClick={() => setShowMermaidPreview(true)}
          >
            Preview Diagram
          </button>
          <Dialog open={showMermaidPreview} onOpenChange={setShowMermaidPreview}>
            <DialogContent>
              <div className="p-6">
                <h3 className="text-lg font-semibold mb-4">Mermaid Diagram Preview</h3>
                {diagram ? (
                  <div dangerouslySetInnerHTML={{ __html: diagram }} />
                ) : (
                  <p className="text-red-500">Unable to render diagram</p>
                )}
              </div>
            </DialogContent>
          </Dialog>
        </div>
      )}
    </div>
  );
}; 