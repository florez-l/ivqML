// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Linear__h__
#define __PUJ__Linear__h__

#include <PUJ/Traits.h>

namespace PUJ
{
  namespace Model
  {
    /**
     */
    template< class _TScalar, class _TTraits = PUJ::Traits< _TScalar > >
    class Linear
    {
    public:
      PUJ_TraitsMacro( Linear );

    public:
      Linear( );
      Linear( const TRow& t );
      Linear( const TRow& w, const TScalar& b );
      virtual ~Linear( ) = default;

      void AnalyticalFit( const TMatrix& X, const TCol& y );

      unsigned long GetDimensions( ) const;
      const TRow& GetWeights( ) const;
      const TScalar& GetBias( ) const;
      const TRow& GetParameters( ) const;

      virtual void Init( unsigned long n, const PUJ::EInitValues& e );

      void SetWeights( const TRow& w );
      void SetBias( const TScalar& b );
      void SetParameters( const TRow& t );

      virtual TScalar operator()( const TRow& x ) const;
      virtual TCol operator()( const TMatrix& x ) const;

    protected:
      void _StreamIn( std::istream& i );
      void _StreamOut( std::ostream& o ) const;

    protected:
      TRow  m_Parameters;

    public:
      /**
       */
      class Cost
      {
      public:
        Cost( Self* model, const TMatrix& X, const TCol& y );
        virtual ~Cost( ) = default;

        TScalar operator()( const TRow& t, TRow* g = nullptr ) const;

      protected:
        Self*   m_Model;
        TMatrix m_XtX;
        TRow    m_uX;
        TRow    m_Xy;
        TScalar m_uy;
        TScalar m_yty;
      };

    public:
      friend std::istream& operator>>( std::istream& i, Self& m )
        {
          m._StreamIn( i );
          return( i );
        }
      friend std::ostream& operator<<( std::ostream& o, const Self& m )
        {
          m._StreamOut( o );
          return( o );
        }
    };
  } // end namespace
} // end namespace

#endif // __PUJ__Linear__h__

// eof - $RCSfile$
